"""Local ESCO candidate matching and conservative, rule-based validation."""
import argparse
import os
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from bson import ObjectId
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pymongo import MongoClient

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')

# These rules concern term/concept pairs, not globally forbidden CV words.
ALIAS_CONFLICTS = {
    ('.net', 'visual basic'),
    ('tensorflow', 'computer vision'),
    ('patterns', 'dies'),
    ('controle d acces', 'supervise security at manned access gates'),
}
COMMON_WORDS = set('avec pour dans sans les des une du de et le la en sur aux the and with for from into of to in a an'.split())


def normalize(text):
    text = unicodedata.normalize('NFKD', (text or '').casefold())
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return re.sub(r'\s+', ' ', text).strip()


def words(text):
    return {w for w in re.findall(r'[^\W\d_]+', normalize(text))
            if len(w) > 3 and w not in COMMON_WORDS}


def split_labels(value):
    if isinstance(value, list):
        return [label for item in value for label in split_labels(item)]
    return [s.strip() for s in re.split(r'\r?\n|\\n', value or '') if s.strip()]


def pair_key(text):
    return re.sub(r"['’]", ' ', normalize(text))


def build_index(rows):
    concepts, terms = {}, defaultdict(set)
    for row in rows:
        uri = row['uri']
        if not uri:
            continue
        concepts[uri] = {
            'uri': uri,
            'label_en': row['label_en'] or '',
            'label_fr': row['label_fr'] or '',
            'description_en': row.get('description_en') or '',
            'description_fr': row.get('description_fr') or '',
        }
        for language in ('en', 'fr'):
            labels = [row[f'label_{language}'] or '']
            labels += split_labels(row.get(f'aliases_{language}'))
            for label in labels:
                term = normalize(label)
                if len(term) >= 3 or term in {'c#', 'c++'}:
                    terms[term].add(uri)
    return concepts, terms


def load_skills():
    query = '''MATCH (s:ESCO:Skill)
    RETURN s.uri AS uri, s.preferredLabel_en AS label_en,
           s.preferredLabel_fr AS label_fr, s.altLabels_en AS aliases_en,
           s.altLabels_fr AS aliases_fr, s.description_en AS description_en,
           s.description_fr AS description_fr'''
    with GraphDatabase.driver(os.environ['NEO4J_URI'], auth=(
        os.environ['NEO4J_USERNAME'], os.environ['NEO4J_PASSWORD']
    )) as driver:
        with driver.session(database=os.getenv('NEO4J_DATABASE', 'neo4j')) as session:
            concepts, terms = build_index(session.run(query).data())
    if not concepts:
        raise ValueError('Aucune competence ESCO dans Neo4j.')
    print(f'Competences ESCO chargees depuis Neo4j : {len(concepts):,}')
    return concepts, terms


def validate_evidence(term, concept, context, before, ambiguous, preferred_elsewhere):
    label = normalize(concept['label_en'])
    if term in COMMON_WORDS:
        return 'rejected', 'common_word_alias'
    if (pair_key(term), label) in ALIAS_CONFLICTS:
        return 'rejected', 'known_alias_conflict'
    # Negation/intention detection is deliberately conservative and incomplete.
    if re.search(r'\b(no|not|without|sans|aucune?|pas|learning|apprendre|souhaite)\b', before):
        return 'uncertain', 'negation_or_learning_context'
    preferred = {normalize(concept[f'label_{lang}']) for lang in ('en', 'fr')}
    if term in preferred and not ambiguous:
        return 'supported', 'unambiguous_preferred_label'
    if preferred_elsewhere and term not in preferred:
        return 'uncertain', 'term_is_another_concepts_preferred_label'
    description = concept['description_en'] + ' ' + concept['description_fr']
    overlap = sorted((words(context) - words(term)) & words(description))
    if overlap:
        return 'uncertain', 'description_overlap_requires_validation: ' + ', '.join(overlap[:8])
    return 'uncertain', 'ambiguous_or_alias_only'


def match_skills(text, concepts, terms):
    normalized = normalize(text)
    matches = defaultdict(list)
    for term, uris in terms.items():
        pattern = rf'(?<!\w){re.escape(term)}(?!\w)'
        occurrences = list(re.finditer(pattern, normalized))
        if not occurrences:
            continue
        preferred_elsewhere = any(
            term == normalize(concepts[uri][f'label_{lang}'])
            for uri in uris for lang in ('en', 'fr')
        )
        for uri in sorted(uris):
            contexts, decisions = [], []
            for occurrence in occurrences:
                context = normalized[max(0, occurrence.start()-120):occurrence.end()+120]
                before = normalized[max(0, occurrence.start()-45):occurrence.start()]
                decision, reason = validate_evidence(
                    term, concepts[uri], context, before,
                    len(uris) > 1, preferred_elsewhere,
                )
                if len(contexts) < 3:
                    contexts.append(context)
                decisions.append({'decision': decision, 'reason': reason})
            status = ('supported' if any(d['decision'] == 'supported' for d in decisions)
                      else 'uncertain' if any(d['decision'] == 'uncertain' for d in decisions)
                      else 'rejected')
            matches[uri].append({
                'matched_term': term, 'ambiguous': len(uris) > 1,
                'contexts': contexts, 'automatic_status': status,
                'reasons': sorted({d['reason'] for d in decisions}),
            })
    results = []
    for uri, evidence in matches.items():
        statuses = {m['automatic_status'] for m in evidence}
        status = ('supported' if 'supported' in statuses else
                  'uncertain' if 'uncertain' in statuses else 'rejected')
        results.append({
            **concepts[uri], 'matches': sorted(evidence, key=lambda m: m['matched_term']),
            'review_status': 'pending',
            'automatic_validation': {'method': 'local_rules_v1', 'decision': status},
        })
    return sorted(results, key=lambda r: r['label_en'].casefold())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cv_id')
    parser.add_argument('--owner', required=True)
    args = parser.parse_args()
    if not ObjectId.is_valid(args.cv_id):
        parser.error('Identifiant CV invalide.')
    required = ('MONGODB_URI', 'MONGODB_DATABASE', 'NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD')
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        parser.error('Configuration manquante : ' + ', '.join(missing))
    identity = {'_id': ObjectId(args.cv_id), 'owner_id': args.owner}
    with MongoClient(os.environ['MONGODB_URI'], serverSelectionTimeoutMS=5000, tz_aware=True) as client:
        cvs = client[os.environ['MONGODB_DATABASE']]['cvs']
        cv = cvs.find_one(identity)
        if cv is None or cv.get('status') != 'extracted':
            raise ValueError('CV absent ou extraction incomplete.')
        expiry = cv.get('expires_at')
        if expiry and expiry <= datetime.now(timezone.utc):
            raise ValueError('CV expire.')
        text = cv.get('text')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Texte du CV vide.')
        concepts, terms = load_skills()
        candidates = match_skills(text, concepts, terms)
        query = {**identity, 'status': 'extracted', 'text': text, 'expires_at': expiry}
        if expiry and expiry <= datetime.now(timezone.utc):
            raise ValueError('CV expire pendant le traitement.')
        result = cvs.update_one(query, {'$set': {'analysis.literal_matching': {
            'method': 'literal_esco_label_match', 'version': 2,
            'validation_method': 'local_rules_v1', 'languages': ['en', 'fr'],
            'generated_at': datetime.now(timezone.utc),
            'candidate_count': len(candidates), 'candidates': candidates,
        }}})
        if result.matched_count != 1:
            raise RuntimeError('CV modifie ou supprime. Relancez la commande.')
    counts = Counter(c['automatic_validation']['decision'] for c in candidates)
    for candidate in candidates:
        print(f"[{candidate['automatic_validation']['decision']}] {candidate['label_en']}")
    print('\nValidation automatique :', dict(counts))
    print('Resultats enregistres dans MongoDB. Revues manuelles conservees.')
    print('Supported = mention explicite selon les regles, pas preuve de maitrise.')


if __name__ == '__main__':
    main()
