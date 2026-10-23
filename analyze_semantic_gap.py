"""Experimental bilingual retrieval + NLI evidence checking, run locally."""
import argparse
import os
import re
from collections import Counter
from datetime import datetime, timezone

from bson import ObjectId
from neo4j import GraphDatabase
from pymongo import MongoClient

# Reuse the existing script's root .env loading, graph query and exact baseline.
from analyze_skill_gap import DATA_SCIENTIST, compare, fetch_requirements

EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
NLI_MODEL = 'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'


def passages(text, size=85, overlap=20):
    """Retain original evidence text and offsets; never join unrelated sections."""
    output = []
    # Bullets, blank paragraphs and sentence endings delimit evidence units.
    boundaries = list(re.finditer(r'[^\n•]+(?:\n(?!\s*\n|\s*[•]))?', text))
    # PDF line wrapping is preserved via adjacent-line windows as well as lines.
    spans = [(m.start(), m.end()) for m in boundaries if m.group().strip()]
    windows = spans + [(spans[i][0], spans[i+1][1]) for i in range(len(spans)-1)
                      if '\n\n' not in text[spans[i][0]:spans[i+1][1]]
                      and '•' not in text[spans[i][1]:spans[i+1][0]]]
    seen = set()
    for start, end in sorted(windows):
        tokens = list(re.finditer(r'\S+', text[start:end]))
        for offset in range(0, len(tokens), size-overlap):
            subset = tokens[offset:offset+size]
            if len(subset) < 3:
                continue
            a, b = start+subset[0].start(), start+subset[-1].end()
            if (a, b) in seen:
                continue
            seen.add((a, b))
            output.append({'start': a, 'end': b, 'text': text[a:b]})
            if offset+size >= len(tokens):
                break
    if not output:
        raise ValueError('Not enough CV text for semantic analysis.')
    return output


def decision_for(review, evidence, threshold):
    if review == 'accepted':
        return 'reviewed_supported'
    if review == 'rejected':
        return 'reviewed_rejected'
    if review == 'uncertain':
        return 'uncertain'
    best = max(evidence, key=lambda e: e['entailment'])
    # Future intentions are not completed experience. Conservative local guard.
    intention = re.search(
        r'\b(objectif|souhaite|en cours|recherche un stage|a la recherche|'
        r'plan to|planning to|intend to|aim to|in progress|seeking)\b',
        best['text'].casefold(),
    )
    if best['entailment'] >= threshold and best['contradiction'] <= 0.1 and not intention:
        return 'model_supported'
    if best['entailment'] >= 0.5 or intention:
        return 'uncertain'
    return 'no_evidence_found'


def describe_requirements(requirements):
    query = '''MATCH (s:ESCO:Skill) WHERE s.uri IN $uris
    RETURN s.uri AS uri, s.description_en AS description_en,
           s.description_fr AS description_fr'''
    with GraphDatabase.driver(os.environ['NEO4J_URI'], auth=(
        os.environ['NEO4J_USERNAME'], os.environ['NEO4J_PASSWORD']
    )) as driver:
        with driver.session(database=os.getenv('NEO4J_DATABASE', 'neo4j')) as session:
            return {r['uri']: dict(r) for r in session.run(query, uris=[r['uri'] for r in requirements])}


def infer(requirements, descriptions, evidence, reviews, threshold, top_k, device):
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print('Loading local models (first use downloads model weights)...', flush=True)
    encoder = SentenceTransformer(EMBEDDING_MODEL, device=device)
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(device).eval()
    labels = {str(v).lower(): int(k) for k, v in model.config.id2label.items()}
    if not {'entailment', 'neutral', 'contradiction'}.issubset(labels):
        raise ValueError(f'Unexpected NLI label mapping: {labels}')

    queries = []
    for requirement in requirements:
        description = descriptions.get(requirement['uri'], {})
        queries.append(' '.join([
            requirement.get('label_en') or '', requirement.get('label_fr') or '',
            (description.get('description_en') or '')[:350],
        ]))
    vectors = encoder.encode([e['text'] for e in evidence], normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=True)
    query_vectors = encoder.encode(queries, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False)
    results = []
    for number, (requirement, vector) in enumerate(zip(requirements, query_vectors), 1):
        similarities = vectors @ vector
        indexes = similarities.argsort()[::-1][:min(top_k, len(evidence))]
        selected = [dict(evidence[int(i)], retrieval_similarity=float(similarities[i])) for i in indexes]
        hypothesis = 'The candidate has demonstrated skill or knowledge in: ' + requirement['label_en'] + '.'
        batch = tokenizer([e['text'] for e in selected], [hypothesis]*len(selected),
                          padding=True, truncation='only_first', max_length=512, return_tensors='pt')
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            probabilities = torch.softmax(model(**batch).logits, dim=-1).cpu().tolist()
        for item, scores in zip(selected, probabilities):
            for label in ('entailment', 'neutral', 'contradiction'):
                item[label] = float(scores[labels[label]])
        selected.sort(key=lambda e: e['entailment'], reverse=True)
        status = decision_for(reviews.get(requirement['uri']), selected, threshold)
        results.append({**requirement, 'decision': status, 'evidence': selected,
                        'hypothesis': hypothesis})
        print(f"[{number}/{len(requirements)}] {status}: {requirement['label_en']}", flush=True)
    return results, {
        'embedding_model': EMBEDDING_MODEL, 'nli_model': NLI_MODEL,
        'embedding_revision': getattr(encoder[0].auto_model.config, '_commit_hash', None),
        'nli_revision': getattr(model.config, '_commit_hash', None),
        'entailment_threshold': threshold, 'top_k': top_k,
        'threshold_calibrated': False,
    }


def summarize(results):
    summary = {}
    for importance in ('essential', 'optional'):
        rows = [r for r in results if r['importance'] == importance]
        counts = Counter(r['decision'] for r in rows)
        total = len(rows)
        supported = counts['reviewed_supported'] + counts['model_supported']
        summary[importance] = {'total': total, 'counts': dict(counts),
            'experimental_coverage_percent': round(100*supported/total, 1) if total else None}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cv_id')
    parser.add_argument('--owner', required=True)
    parser.add_argument('--occupation-uri', default=DATA_SCIENTIST)
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if not ObjectId.is_valid(args.cv_id):
        parser.error('Invalid CV ID.')
    if not 0.5 < args.threshold <= 1 or not 1 <= args.top_k <= 10:
        parser.error('Threshold must be > 0.5 and <= 1; top-k must be 1..10.')
    required = ('MONGODB_URI','MONGODB_DATABASE','NEO4J_URI','NEO4J_USERNAME','NEO4J_PASSWORD')
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        parser.error('Missing .env settings: ' + ', '.join(missing))
    identity = {'_id': ObjectId(args.cv_id), 'owner_id': args.owner}
    with MongoClient(os.environ['MONGODB_URI'], serverSelectionTimeoutMS=5000, tz_aware=True) as client:
        cvs = client[os.environ['MONGODB_DATABASE']]['cvs']
        cv = cvs.find_one(identity)
        if not cv or cv.get('status') != 'extracted':
            raise ValueError('CV absent or not ready.')
        expiry = cv.get('expires_at')
        if expiry and expiry <= datetime.now(timezone.utc):
            raise ValueError('CV expired.')
        text = cv.get('text')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('CV text is empty.')
        analysis = cv.get('analysis', {})
        reviews = analysis.get('manual_review', [])
        if not isinstance(reviews, list):
            raise ValueError('Invalid manual review format.')
        occupation, raw = fetch_requirements(args.occupation_uri)
        baseline = compare(raw, reviews)  # Also validates reviews and relationships.
        unique = {}
        for row in raw:
            if row['uri'] not in unique or row['importance'] == 'essential':
                unique[row['uri']] = {k: row.get(k) for k in ('uri','label_en','label_fr','importance')}
        requirements = list(unique.values())
        descriptions = describe_requirements(requirements)
        results, config = infer(requirements, descriptions, passages(text),
            {r['uri']: r['decision'] for r in reviews}, args.threshold, args.top_k, args.device)
        summary = summarize(results)
        if expiry and expiry <= datetime.now(timezone.utc):
            raise ValueError('CV expired during processing.')
        saved = cvs.update_one({**identity, 'status': 'extracted', 'text': text, 'expires_at': expiry,
            'analysis.manual_review': reviews if 'manual_review' in analysis else {'$exists': False}},
            {'$set': {'analysis.semantic_skill_gap': {
                'version': 1, 'occupation': occupation, 'generated_at': datetime.now(timezone.utc),
                'config': config, 'review_snapshot': reviews, 'exact_baseline': baseline,
                'requirements': results, 'summary': summary,
                'note': 'Experimental model-assisted coverage, not employability or mastery; NLI scores are uncalibrated.',
            }}})
        if saved.matched_count != 1:
            raise RuntimeError('CV or reviews changed. Rerun.')
    for importance, group in summary.items():
        print(f"\n{importance.upper()}: exact baseline {baseline[importance]['coverage_percent']}%")
        print(f"Experimental model-assisted coverage: {group['experimental_coverage_percent']}%")
        print(group['counts'])
    print('\nSaved: analysis.semantic_skill_gap. Exact baseline and manual reviews preserved.')
    print('Inspect model-supported evidence before using results for recommendations.')


if __name__ == '__main__':
    main()
