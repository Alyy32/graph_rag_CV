# graph_rag_CV
L'étudiant importe son CV : ses compétences sont extraites et comparées à celles requises par le métier choisi. Le système identifie les compétences manquantes, propose les cours correspondants et explique chaque recommandation. Durée du projet : 4 mois (préparation des données, modélisation du graphe, entraînement de l'IA, puis tests).

## Structure

- `src/cv_pipeline/pdf.py` : extraction PDF partagée par tous les flux.
- `src/cv_pipeline/ingest.py` : ingestion du CV en mémoire vers MongoDB/GridFS.
- `extract_cv.py` : script de compatibilité pour l'extraction locale et la détection de langue.
- `data/` : fichiers source ESCO et résultats générés.

## Ingestion MongoDB

```powershell
python -m src.cv_pipeline.ingest path\to\cv.pdf --owner owner-id
```
