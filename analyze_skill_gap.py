"""Compare manually reviewed CV skills with direct ESCO occupation requirements."""
import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from bson import ObjectId
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pymongo import MongoClient

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
DATA_SCIENTIST = 'http://data.europa.eu/esco/occupation/258e46f9-0075-4a2e-adae-1ff0477e0f30'


def compare(requirements, reviews):
    """Exact URI comparison; duplicate relations count once, essential takes precedence."""
    decisions = {}
    for review in reviews:
        uri, decision = review.get('uri'), review.get('decision')
        if not uri or decision not in {'accepted', 'rejected', 'uncertain'}:
            raise ValueError('Invalid manual review entry.')
        if uri in decisions and decisions[uri] != decision:
            raise ValueError(f'Conflicting reviews for {uri}.')
        decisions[uri] = decision

    unique = {}
    for requirement in requirements:
        uri = requirement.get('uri')
        importance = requirement.get('importance')
        if not uri or importance not in {'essential', 'optional'}:
            raise ValueError('Invalid ESCO requirement URI or importance.')
        if uri not in unique or importance == 'essential':
            unique[uri] = requirement

    groups = {
        importance: {'matched': [], 'uncertain': [], 'not_evidenced': []}
        for importance in ('essential', 'optional')
    }
    for uri, requirement in unique.items():
        decision = decisions.get(uri)
        category = ('matched' if decision == 'accepted' else
                    'uncertain' if decision == 'uncertain' else 'not_evidenced')
        groups[requirement['importance']][category].append({
            'uri': uri,
            'label_en': requirement.get('label_en') or '',
            'label_fr': requirement.get('label_fr') or '',
        })

    for group in groups.values():
        for entries in group.values():
            entries.sort(key=lambda item: item['label_en'].casefold())
        total = sum(len(entries) for entries in group.values())
        group['total'] = total
        group['coverage_percent'] = (
            round(100 * len(group['matched']) / total, 1) if total else None
        )
    return groups


def fetch_requirements(occupation_uri):
    query = '''
    MATCH (o:ESCO:Occupation {uri: $uri})
    OPTIONAL MATCH (o)-[r:REQUIRES_SKILL]->(s:ESCO:Skill)
    RETURN o.uri AS occupation_uri,
           o.preferredLabel_en AS occupation_en,
           o.preferredLabel_fr AS occupation_fr,
           s.uri AS uri, s.preferredLabel_en AS label_en,
           s.preferredLabel_fr AS label_fr, r.importance AS importance
    '''
    with GraphDatabase.driver(os.environ['NEO4J_URI'], auth=(
        os.environ['NEO4J_USERNAME'], os.environ['NEO4J_PASSWORD']
    )) as driver:
        with driver.session(database=os.getenv('NEO4J_DATABASE', 'neo4j')) as session:
            rows = session.run(query, uri=occupation_uri).data()
    if not rows:
        raise ValueError('Occupation not found in Neo4j. Check its URI.')
    requirements = [row for row in rows if row['uri'] is not None]
    if not requirements:
        raise ValueError('No requirements found. Check the ESCO relationship import.')
    occupation = {
        'uri': rows[0]['occupation_uri'],
        'label_en': rows[0]['occupation_en'] or '',
        'label_fr': rows[0]['occupation_fr'] or '',
    }
    return occupation, requirements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cv_id')
    parser.add_argument('--owner', required=True)
    parser.add_argument('--occupation-uri', default=DATA_SCIENTIST)
    args = parser.parse_args()
    if not ObjectId.is_valid(args.cv_id):
        parser.error('Invalid CV ID.')
    keys = ('MONGODB_URI', 'MONGODB_DATABASE', 'NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD')
    missing = [key for key in keys if not os.getenv(key)]
    if missing:
        parser.error('Missing configuration: ' + ', '.join(missing))

    identity = {'_id': ObjectId(args.cv_id), 'owner_id': args.owner}
    with MongoClient(os.environ['MONGODB_URI'], serverSelectionTimeoutMS=5000, tz_aware=True) as client:
        cvs = client[os.environ['MONGODB_DATABASE']]['cvs']
        cv = cvs.find_one(identity)
        if not cv or cv.get('status') != 'extracted':
            raise ValueError('CV not found for this owner or not ready.')
        expiry = cv.get('expires_at')
        if expiry and expiry <= datetime.now(timezone.utc):
            raise ValueError('This CV has expired.')
        reviews = cv.get('analysis', {}).get('manual_review')
        if not isinstance(reviews, list) or not reviews:
            raise ValueError('No manual reviews found. Run review_cv_skills.py first.')

        occupation, requirements = fetch_requirements(args.occupation_uri)
        groups = compare(requirements, reviews)
        result = {
            'method': 'reviewed_esco_uri_comparison',
            'version': 1,
            'generated_at': datetime.now(timezone.utc),
            'occupation': occupation,
            'review_snapshot': reviews,
            'comparison': groups,
            'interpretation': 'Coverage of direct ESCO requirements evidenced in this CV; not employability or mastery.',
        }
        if expiry and expiry <= datetime.now(timezone.utc):
            raise ValueError('CV expired during processing.')
        # Store the latest comparison only; preserve matching and manual reviews.
        saved = cvs.update_one({
            **identity, 'status': 'extracted', 'expires_at': expiry,
            'text': cv.get('text'), 'analysis.manual_review': reviews,
        }, {'$set': {'analysis.skill_gap': result}})
        if saved.matched_count != 1:
            raise RuntimeError('CV or reviews changed during processing. Run again.')

    print(f"\nTarget: {occupation['label_en']} | {occupation['label_fr']}")
    for importance, group in groups.items():
        coverage = group['coverage_percent']
        print(f"\n{importance.upper()}: {group['total']} requirements")
        print(f"Coverage: {coverage}%" if coverage is not None else 'Coverage: N/A')
        for category in ('matched', 'uncertain', 'not_evidenced'):
            print(f"\n  {category.replace('_', ' ').title()}: {len(group[category])}")
            for skill in group[category]:
                print(f"    - {skill['label_en']} | {skill['label_fr']}")
    print('\nSaved in MongoDB: analysis.skill_gap')
    print('Not evidenced means absent from the accepted CV evidence, not necessarily a skill you lack.')
    print('Exact URI comparison only; related or broader skills are not inferred.')


if __name__ == '__main__':
    main()
