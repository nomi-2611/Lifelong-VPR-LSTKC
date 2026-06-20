from __future__ import annotations

import importlib


MODULES = [
    "src.datasets.lreid_dataset.datasets",
    "src.models.reid_models.resnet",
    "src.models.reid_models.vprtempo_backbone",
    "src.trainers.trainer",
    "src.retrieval.tools.eval_stage_expert_pool",
]


def main() -> None:
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        print(f"OK {module_name}: {module.__file__}")


if __name__ == "__main__":
    main()
