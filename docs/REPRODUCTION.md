# Reproduction Notes

The main continual training entry is:

```text
src/trainers/continual_train.py
```

The key thesis scripts are kept under:

```text
scripts/
```

Before running, update:

- data root;
- logs directory;
- VPRTempo pretrained model path;
- preprocessing cache path.

The projected descriptor protocol is used for the long-short-term knowledge consolidation analysis. Stage expert and sequence-consistency retrieval are evaluated using tools under:

```text
src/retrieval/tools/
```
