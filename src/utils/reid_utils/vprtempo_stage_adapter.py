from __future__ import absolute_import

import csv
import os
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import sys


VPRTEMPO_ROOT = Path(__file__).resolve().parents[2] / "feature_extraction" / "vprtempo_snn"
if VPRTEMPO_ROOT.exists() and str(VPRTEMPO_ROOT) not in sys.path:
    sys.path.insert(0, str(VPRTEMPO_ROOT))

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from vprtempo.VPRTempoTrain import VPRTempoTrain
from vprtempo.src.dataset import CustomImageDataset, ProcessImage


def _sanitize_name(name):
    return ''.join(ch if ch.isalnum() else '_' for ch in str(name))


def _infer_model_structure(model_path):
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model_keys = sorted(state.keys(), key=lambda key: int(key.split("_")[1]))
    out_dims = [int(state[key]["output_layer.w.weight"].shape[0]) for key in model_keys]
    return {
        "num_modules": len(model_keys),
        "out_dims": out_dims,
        "database_places": int(sum(out_dims)),
        "max_module": int(max(out_dims)),
        "out_dim": int(out_dims[0]),
        "final_out_dim": int(out_dims[-1]),
    }


def _maybe_load_initial_weights(models, model_path):
    if not model_path:
        return
    combined_state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    available_keys = sorted(combined_state_dict.keys())
    for i, model in enumerate(models):
        state_key = "model_{}".format(i)
        if state_key not in combined_state_dict:
            state_key = available_keys[min(i, len(available_keys) - 1)]
        source_state = combined_state_dict[state_key]
        target_state = model.state_dict()
        filtered_state = {
            key: value
            for key, value in source_state.items()
            if key in target_state and target_state[key].shape == value.shape
        }
        model.load_state_dict(filtered_state, strict=False)


def _group_samples_for_stage(dataset, data_root, max_pid=None):
    root = Path(data_root).resolve()
    groups = defaultdict(list)
    for sample in dataset.train:
        fpath, pid, _, _ = sample
        pid = int(pid)
        if max_pid is not None and pid >= max_pid:
            continue
        rel_path = str(Path(fpath).resolve().relative_to(root)).replace("\\", "/")
        parent_rel = str(Path(rel_path).parent).replace("\\", "/")
        groups[parent_rel].append((rel_path, pid))
    ordered = {}
    for parent_rel, samples in sorted(groups.items()):
        ordered[parent_rel] = sorted(samples, key=lambda item: (item[1], item[0]))
    return ordered


def _write_stage_csvs(dataset_name, grouped_samples, output_dir, stage_tag):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_keys = []
    for parent_rel, samples in grouped_samples.items():
        key = "{}_{}".format(stage_tag, _sanitize_name(parent_rel))
        csv_path = output_dir / "{}-{}.csv".format(dataset_name, key)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Image_name", "index"])
            for rel_path, pid in samples:
                writer.writerow([rel_path, pid])
        csv_keys.append(key)
    return csv_keys


def _build_models(stage_args, dims, structure):
    models = []
    final_out = None
    for mod in range(structure["num_modules"]):
        model = VPRTempoTrain(
            stage_args,
            dims,
            logger=None,
            num_modules=structure["num_modules"],
            out_dim=structure["out_dim"],
            out_dim_remainder=final_out,
        )
        model.to(torch.device("cpu"))
        models.append(model)
        if mod == structure["num_modules"] - 2 and structure["final_out_dim"] != structure["out_dim"]:
            final_out = structure["final_out_dim"]
    return models


def _train_stage_models(models, output_path):
    image_transform = transforms.Compose([
        ProcessImage(models[0].dims, models[0].patches)
    ])
    user_input_ranges = []
    start_idx = 0
    for _ in range(models[0].num_modules):
        range_temp = [start_idx, start_idx + ((models[0].max_module - 1) * models[0].filter)]
        user_input_ranges.append(range_temp)
        start_idx = range_temp[1] + models[0].filter

    trained_layers = []
    train_layers = getattr(models[0], "train_layers", "all")
    prev_cwd = os.getcwd()
    os.chdir(str(VPRTEMPO_ROOT))
    try:
        for layer_name, _ in sorted(models[0].layer_dict.items(), key=lambda item: item[1]):
            if train_layers == "output_only" and layer_name != "output_layer":
                trained_layers.append(layer_name)
                continue
            if train_layers == "output_plus_last_feature" and layer_name not in {"feature_layer", "output_layer"}:
                trained_layers.append(layer_name)
                continue

            for i, model in enumerate(models):
                model.train()
                model.to(torch.device(model.device))
                layer = getattr(model, layer_name)
                if model.database_places < model.max_module:
                    max_samples = model.database_places
                elif model.output < model.max_module:
                    max_samples = model.output
                else:
                    max_samples = model.max_module
                img_range = user_input_ranges[i]
                train_dataset = CustomImageDataset(
                    annotations_file=models[0].dataset_file,
                    base_dir=models[0].data_dir,
                    img_dirs=models[0].database_dirs,
                    transform=image_transform,
                    filter=models[0].filter,
                    skip=models[0].skip,
                    test=False,
                    img_range=img_range,
                    max_samples=max_samples,
                )
                if len(train_dataset) == 0:
                    model.to(torch.device("cpu"))
                    continue
                if model.device == "mps" or os.name == "nt":
                    num_workers = 0
                elif hasattr(model, "workers"):
                    num_workers = model.workers
                else:
                    num_workers = 4
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=1,
                    shuffle=True,
                    num_workers=num_workers,
                    persistent_workers=False,
                )
                model.train_model(train_loader, layer, model, i, prev_layers=trained_layers)
                model.to(torch.device("cpu"))
            trained_layers.append(layer_name)
    finally:
        os.chdir(prev_cwd)

    for model in models:
        model.eval()
    models[0].save_model(models, str(output_path))


def adapt_vprtempo_for_stage(args, dataset, stage_name, stage_index, current_model_path):
    if current_model_path is None:
        raise ValueError("Stage adaptation requires a base VPRTempo model path")

    structure = _infer_model_structure(current_model_path)
    grouped_samples = _group_samples_for_stage(
        dataset,
        data_root=args.data_dir,
        max_pid=structure["database_places"],
    )
    if not grouped_samples:
        return current_model_path

    stage_tag = "stage{:02d}_{}".format(stage_index + 1, _sanitize_name(stage_name))
    dataset_name = getattr(args, "vprtempo_stage_dataset_name", "place_stageadapt")
    csv_output_dir = VPRTEMPO_ROOT / "vprtempo" / "dataset"
    csv_keys = _write_stage_csvs(dataset_name, grouped_samples, csv_output_dir, stage_tag)

    dims = [int(args.vprtempo_input_h), int(args.vprtempo_input_w)]
    stage_args = SimpleNamespace(
        dataset=dataset_name,
        data_dir=args.data_dir,
        database_places=structure["database_places"],
        query_places=0,
        max_module=structure["max_module"],
        database_dirs=",".join(csv_keys),
        database_img_dirs=",".join(["."] * len(csv_keys)),
        query_dir="",
        GT_tolerance=0,
        skip=0,
        filter=args.vprtempo_stage_filter,
        epoch=args.vprtempo_stage_epochs,
        workers=args.workers,
        train_layers=args.vprtempo_stage_train_layers,
        feature_ip_rate=args.vprtempo_stage_feature_ip_rate,
        feature_stdp_rate=args.vprtempo_stage_feature_stdp_rate,
        output_ip_rate=args.vprtempo_stage_output_ip_rate,
        output_stdp_rate=args.vprtempo_stage_output_stdp_rate,
        patches=args.vprtempo_patches,
        dims="{},{}".format(args.vprtempo_input_h, args.vprtempo_input_w),
    )
    models = _build_models(stage_args, dims, structure)
    _maybe_load_initial_weights(models, current_model_path)

    output_dir = Path(args.vprtempo_stage_output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "{}.pth".format(stage_tag)
    if output_path.exists() and not args.vprtempo_stage_force_retrain:
        return str(output_path)

    _train_stage_models(models, output_path)
    return str(output_path)


