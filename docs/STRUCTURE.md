# Code Structure

This refactored layout maps thesis chapters to source folders.

| Thesis Part | Folder |
|---|---|
| Multi-environment sequential VPR task | `src/datasets/` |
| VPRTempo-SNN feature extraction | `src/feature_extraction/vprtempo_snn/`, `src/models/reid_models/vprtempo_backbone.py` |
| Descriptor and relation matrix construction | `src/knowledge/` |
| Short-term knowledge transfer | `src/trainers/trainer.py`, `src/knowledge/` |
| Long-term knowledge retention | `src/trainers/trainer.py`, `src/knowledge/` |
| Stage expert pool and sequence retrieval | `src/retrieval/tools/` |
| Evaluation metrics | `src/evaluation/` |

The code now uses explicit `src.*` imports. Core implementation is kept under `src/`, so no root-level compatibility packages are required.

VPRTempo-SNN is integrated under `src/feature_extraction/vprtempo_snn/`. In this project it serves as the feature extraction module for lifelong visual place recognition, rather than being kept as a separate third-party repository.
