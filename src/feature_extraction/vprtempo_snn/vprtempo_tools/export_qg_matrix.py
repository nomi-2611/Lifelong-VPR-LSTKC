import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vprtempo.VPRTempo import VPRTempo
from vprtempo.src.dataset import CustomImageDataset, ProcessImage
from vprtempo.src.loggers import model_logger


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder):
    folder = Path(folder)
    files = [p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    return sorted(files)


def create_csv(csv_path, image_names):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write("Image_name,index\n")
        for idx, name in enumerate(image_names):
            f.write(f"{name},{idx}\n")


class ExplicitPathImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = [str(Path(p)) for p in image_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, idx


def filtered_count(image_names, skip, filt, max_samples):
    selected = image_names[skip::filt]
    if max_samples is not None:
        selected = selected[:max_samples]
    return len(selected)


def infer_model_structure(model_path):
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model_keys = sorted(state.keys(), key=lambda x: int(x.split("_")[1]))
    out_dims = [state[key]["output_layer.w.weight"].shape[0] for key in model_keys]
    return {
        "num_modules": len(model_keys),
        "out_dims": out_dims,
        "database_places": int(sum(out_dims)),
        "out_dim": int(out_dims[0]),
        "final_out_dim": int(out_dims[-1]),
    }


def build_models(args, dims, logger, output_folder, structure):
    models = []
    final_out = None
    for mod in range(structure["num_modules"]):
        model = VPRTempo(
            args,
            dims,
            logger,
            structure["num_modules"],
            output_folder,
            structure["out_dim"],
            out_dim_remainder=final_out,
        )
        model.eval()
        model.to(torch.device("cpu"))
        models.append(model)
        if mod == structure["num_modules"] - 2 and structure["final_out_dim"] != structure["out_dim"]:
            final_out = structure["final_out_dim"]
    return models


def load_lstkc_split_paths(dataset_name, data_root, split_name, repo_root):
    repo_root = Path(repo_root).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import src.datasets.lreid_dataset.datasets as lstkc_datasets

    dataset = lstkc_datasets.create(dataset_name, str(Path(data_root).resolve()))
    if not hasattr(dataset, split_name):
        raise AttributeError(f"LSTKC dataset {dataset_name} does not have split {split_name}")
    split = getattr(dataset, split_name)
    return [str(Path(sample[0]).resolve()) for sample in split]


def export_qg(args):
    structure = infer_model_structure(args.model_path)
    query_folder = Path(args.data_dir) / args.query_dir
    if not query_folder.exists():
        raise FileNotFoundError(f"Query folder not found: {query_folder}")

    query_images = list_images(query_folder)
    if not query_images:
        raise FileNotFoundError(f"No supported images found in {query_folder}")

    dataset_key = f"{args.dataset}-{args.query_dir}"
    csv_path = ROOT / "vprtempo" / "dataset" / f"{dataset_key}.csv"
    create_csv(csv_path, query_images)
    query_places = filtered_count(query_images, args.skip, args.filter, args.query_places)

    runtime_args = SimpleNamespace(
        dataset=args.dataset,
        data_dir=args.data_dir,
        database_places=structure["database_places"],
        query_places=query_places,
        max_module=max(structure["out_dims"]),
        database_dirs=args.database_dirs,
        query_dir=args.query_dir or "query",
        GT_tolerance=args.gt_tolerance,
        skip=args.skip,
        filter=args.filter,
        epoch=1,
        patches=args.patches,
        dims=",".join(str(x) for x in args.dims),
        train_new_model=False,
        quantize=False,
        PR_curve=False,
        sim_mat=False,
        run_demo=False,
        export_matrix=True,
        export_prefix=args.export_prefix,
    )

    logger, output_folder = model_logger()
    models = build_models(runtime_args, args.dims, logger, output_folder, structure)
    image_transform = ProcessImage(args.dims, args.patches)
    test_dataset = CustomImageDataset(
        annotations_file=str(csv_path),
        base_dir=args.data_dir,
        img_dirs=[args.query_dir],
        transform=image_transform,
        max_samples=args.query_places,
        filter=args.filter,
        skip=args.skip,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.workers,
        persistent_workers=False,
    )
    models[0].load_model(models, args.model_path)
    with torch.no_grad():
        models[0].evaluate(models, test_loader)

    output_folder = Path(output_folder)
    similarity_path = output_folder / f"{args.export_prefix}_similarity.npy"
    gt_path = output_folder / f"{args.export_prefix}_gt.npy"

    final_output_dir = Path(args.output_dir) if args.output_dir else output_folder
    final_output_dir.mkdir(parents=True, exist_ok=True)
    final_similarity = final_output_dir / f"{args.export_prefix}_qg.npy"
    final_gt = final_output_dir / f"{args.export_prefix}_gt.npy"
    if similarity_path.resolve() != final_similarity.resolve():
        shutil.copy2(similarity_path, final_similarity)
    if gt_path.exists() and gt_path.resolve() != final_gt.resolve():
        shutil.copy2(gt_path, final_gt)

    manifest = {
        args.manifest_dataset_name: {
            "qg": str(final_similarity.resolve()).replace("\\", "/"),
            "kind": "similarity",
        }
    }
    manifest_path = final_output_dir / f"{args.export_prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary = {
        "model_path": str(Path(args.model_path).resolve()).replace("\\", "/"),
        "query_dir": str(query_folder.resolve()).replace("\\", "/"),
        "database_places": structure["database_places"],
        "query_places": query_places,
        "qg_path": str(final_similarity.resolve()).replace("\\", "/"),
        "manifest_path": str(manifest_path.resolve()).replace("\\", "/"),
    }
    summary_path = final_output_dir / f"{args.export_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def extract_embeddings(models, dataset, workers):
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=workers,
        persistent_workers=False,
    )
    model = models[0]
    model.inferences = []
    for sub_model in models:
        model.inferences.append(
            torch.nn.Sequential(
                sub_model.feature_layer.w,
                sub_model.output_layer.w,
            ).to(torch.device(model.device))
        )
    embeddings = []
    with torch.no_grad():
        for spikes, _ in loader:
            spikes = spikes.to(model.device)
            out = model.forward(spikes)
            embeddings.append(out.detach().cpu().squeeze(0))
    return torch.stack(embeddings, dim=0)


def export_qg_from_query_gallery(args):
    structure = infer_model_structure(args.model_path)
    image_transform = ProcessImage(args.dims, args.patches)

    if args.lstkc_dataset_name:
        query_paths = load_lstkc_split_paths(
            dataset_name=args.lstkc_dataset_name,
            data_root=args.lstkc_data_root,
            split_name="query",
            repo_root=args.lstkc_repo_root,
        )
        gallery_paths = load_lstkc_split_paths(
            dataset_name=args.lstkc_dataset_name,
            data_root=args.lstkc_data_root,
            split_name="gallery",
            repo_root=args.lstkc_repo_root,
        )

        query_paths = query_paths[args.skip_query::args.filter_query]
        gallery_paths = gallery_paths[args.skip_gallery::args.filter_gallery]
        if args.query_places is not None:
            query_paths = query_paths[:args.query_places]
        if args.gallery_places is not None:
            gallery_paths = gallery_paths[:args.gallery_places]

        if not query_paths or not gallery_paths:
            raise FileNotFoundError("Resolved LSTKC query/gallery split is empty after filtering")

        query_places = len(query_paths)
        gallery_places = len(gallery_paths)
        query_dataset = ExplicitPathImageDataset(query_paths, transform=image_transform)
        gallery_dataset = ExplicitPathImageDataset(gallery_paths, transform=image_transform)
        query_source = args.lstkc_dataset_name
        gallery_source = args.lstkc_dataset_name
    else:
        query_folder = Path(args.data_dir) / args.query_dir
        gallery_folder = Path(args.data_dir) / args.gallery_dir
        if not query_folder.exists():
            raise FileNotFoundError(f"Query folder not found: {query_folder}")
        if not gallery_folder.exists():
            raise FileNotFoundError(f"Gallery folder not found: {gallery_folder}")

        query_images = list_images(query_folder)
        gallery_images = list_images(gallery_folder)
        if not query_images or not gallery_images:
            raise FileNotFoundError("Query or gallery folders do not contain supported image files")

        query_csv = ROOT / "vprtempo" / "dataset" / f"{args.dataset}-{args.query_dir}.csv"
        gallery_csv = ROOT / "vprtempo" / "dataset" / f"{args.dataset}-{args.gallery_dir}.csv"
        create_csv(query_csv, query_images)
        create_csv(gallery_csv, gallery_images)

        query_places = filtered_count(query_images, args.skip_query, args.filter_query, args.query_places)
        gallery_places = filtered_count(gallery_images, args.skip_gallery, args.filter_gallery, args.gallery_places)

        query_dataset = CustomImageDataset(
            annotations_file=str(query_csv),
            base_dir=args.data_dir,
            img_dirs=[args.query_dir],
            transform=image_transform,
            max_samples=args.query_places,
            filter=args.filter_query,
            skip=args.skip_query,
        )
        gallery_dataset = CustomImageDataset(
            annotations_file=str(gallery_csv),
            base_dir=args.data_dir,
            img_dirs=[args.gallery_dir],
            transform=image_transform,
            max_samples=args.gallery_places,
            filter=args.filter_gallery,
            skip=args.skip_gallery,
        )
        query_source = str(query_folder.resolve()).replace("\\", "/")
        gallery_source = str(gallery_folder.resolve()).replace("\\", "/")

    runtime_args = SimpleNamespace(
        dataset=args.dataset,
        data_dir=args.data_dir,
        database_places=structure["database_places"],
        query_places=query_places,
        max_module=max(structure["out_dims"]),
        database_dirs=args.database_dirs,
        query_dir=args.query_dir or "query",
        GT_tolerance=0,
        skip=0,
        filter=1,
        epoch=1,
        patches=args.patches,
        dims=",".join(str(x) for x in args.dims),
        train_new_model=False,
        quantize=False,
        PR_curve=False,
        sim_mat=False,
        run_demo=False,
        export_matrix=False,
        export_prefix=args.export_prefix,
    )

    logger, output_folder = model_logger()
    models = build_models(runtime_args, args.dims, logger, output_folder, structure)
    models[0].load_model(models, args.model_path)

    query_embeddings = extract_embeddings(models, query_dataset, args.workers)
    gallery_embeddings = extract_embeddings(models, gallery_dataset, args.workers)
    query_embeddings = F.normalize(query_embeddings, dim=1)
    gallery_embeddings = F.normalize(gallery_embeddings, dim=1)
    qg_similarity = torch.mm(query_embeddings, gallery_embeddings.t()).cpu().numpy().astype(np.float32)

    final_output_dir = Path(args.output_dir)
    final_output_dir.mkdir(parents=True, exist_ok=True)
    qg_path = final_output_dir / f"{args.export_prefix}_qg.npy"
    np.save(qg_path, qg_similarity)

    manifest = {
        args.manifest_dataset_name: {
            "qg": str(qg_path.resolve()).replace("\\", "/"),
            "kind": "similarity",
        }
    }
    manifest_path = final_output_dir / f"{args.export_prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary = {
        "model_path": str(Path(args.model_path).resolve()).replace("\\", "/"),
        "query_dir": query_source,
        "gallery_dir": gallery_source,
        "query_places": query_places,
        "gallery_places": gallery_places,
        "qg_shape": [int(qg_similarity.shape[0]), int(qg_similarity.shape[1])],
        "qg_path": str(qg_path.resolve()).replace("\\", "/"),
        "manifest_path": str(manifest_path.resolve()).replace("\\", "/"),
    }
    summary_path = final_output_dir / f"{args.export_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def export_embeddings_from_query_gallery(args):
    structure = infer_model_structure(args.model_path)
    image_transform = ProcessImage(args.dims, args.patches)

    if args.lstkc_dataset_name:
        query_paths = load_lstkc_split_paths(
            dataset_name=args.lstkc_dataset_name,
            data_root=args.lstkc_data_root,
            split_name="query",
            repo_root=args.lstkc_repo_root,
        )
        gallery_paths = load_lstkc_split_paths(
            dataset_name=args.lstkc_dataset_name,
            data_root=args.lstkc_data_root,
            split_name="gallery",
            repo_root=args.lstkc_repo_root,
        )

        query_paths = query_paths[args.skip_query::args.filter_query]
        gallery_paths = gallery_paths[args.skip_gallery::args.filter_gallery]
        if args.query_places is not None:
            query_paths = query_paths[:args.query_places]
        if args.gallery_places is not None:
            gallery_paths = gallery_paths[:args.gallery_places]

        if not query_paths or not gallery_paths:
            raise FileNotFoundError("Resolved LSTKC query/gallery split is empty after filtering")

        query_places = len(query_paths)
        gallery_places = len(gallery_paths)
        query_dataset = ExplicitPathImageDataset(query_paths, transform=image_transform)
        gallery_dataset = ExplicitPathImageDataset(gallery_paths, transform=image_transform)
        query_source = args.lstkc_dataset_name
        gallery_source = args.lstkc_dataset_name
    else:
        if not args.query_dir or not args.gallery_dir:
            raise ValueError("Both --query-dir and --gallery-dir are required for embedding export without --lstkc-dataset-name")
        query_folder = Path(args.data_dir) / args.query_dir
        gallery_folder = Path(args.data_dir) / args.gallery_dir
        if not query_folder.exists():
            raise FileNotFoundError(f"Query folder not found: {query_folder}")
        if not gallery_folder.exists():
            raise FileNotFoundError(f"Gallery folder not found: {gallery_folder}")

        query_images = list_images(query_folder)
        gallery_images = list_images(gallery_folder)
        if not query_images or not gallery_images:
            raise FileNotFoundError("Query or gallery folders do not contain supported image files")

        query_paths = [str((query_folder / name).resolve()) for name in query_images]
        gallery_paths = [str((gallery_folder / name).resolve()) for name in gallery_images]
        query_paths = query_paths[args.skip_query::args.filter_query]
        gallery_paths = gallery_paths[args.skip_gallery::args.filter_gallery]
        if args.query_places is not None:
            query_paths = query_paths[:args.query_places]
        if args.gallery_places is not None:
            gallery_paths = gallery_paths[:args.gallery_places]

        query_places = len(query_paths)
        gallery_places = len(gallery_paths)
        query_dataset = ExplicitPathImageDataset(query_paths, transform=image_transform)
        gallery_dataset = ExplicitPathImageDataset(gallery_paths, transform=image_transform)
        query_source = str(query_folder.resolve()).replace("\\", "/")
        gallery_source = str(gallery_folder.resolve()).replace("\\", "/")

    runtime_args = SimpleNamespace(
        dataset=args.dataset,
        data_dir=args.data_dir,
        database_places=structure["database_places"],
        query_places=query_places,
        max_module=max(structure["out_dims"]),
        database_dirs=args.database_dirs,
        query_dir=args.query_dir or "query",
        GT_tolerance=0,
        skip=0,
        filter=1,
        epoch=1,
        patches=args.patches,
        dims=",".join(str(x) for x in args.dims),
        train_new_model=False,
        quantize=False,
        PR_curve=False,
        sim_mat=False,
        run_demo=False,
        export_matrix=False,
        export_prefix=args.export_prefix,
    )

    logger, output_folder = model_logger()
    models = build_models(runtime_args, args.dims, logger, output_folder, structure)
    models[0].load_model(models, args.model_path)

    query_embeddings = extract_embeddings(models, query_dataset, args.workers)
    gallery_embeddings = extract_embeddings(models, gallery_dataset, args.workers)
    query_embeddings = F.normalize(query_embeddings, dim=1).cpu().numpy().astype(np.float32)
    gallery_embeddings = F.normalize(gallery_embeddings, dim=1).cpu().numpy().astype(np.float32)

    final_output_dir = Path(args.output_dir)
    final_output_dir.mkdir(parents=True, exist_ok=True)
    query_emb_path = final_output_dir / f"{args.export_prefix}_query_embeddings.npy"
    gallery_emb_path = final_output_dir / f"{args.export_prefix}_gallery_embeddings.npy"
    np.save(query_emb_path, query_embeddings)
    np.save(gallery_emb_path, gallery_embeddings)

    manifest = {
        args.manifest_dataset_name: {
            "query_embeddings": str(query_emb_path.resolve()).replace("\\", "/"),
            "gallery_embeddings": str(gallery_emb_path.resolve()).replace("\\", "/"),
            "metric": "cosine",
            "normalized": True,
        }
    }
    manifest_path = final_output_dir / f"{args.export_prefix}_embedding_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary = {
        "model_path": str(Path(args.model_path).resolve()).replace("\\", "/"),
        "query_dir": query_source,
        "gallery_dir": gallery_source,
        "query_places": query_places,
        "gallery_places": gallery_places,
        "query_embedding_shape": [int(query_embeddings.shape[0]), int(query_embeddings.shape[1])],
        "gallery_embedding_shape": [int(gallery_embeddings.shape[0]), int(gallery_embeddings.shape[1])],
        "query_embeddings_path": str(query_emb_path.resolve()).replace("\\", "/"),
        "gallery_embeddings_path": str(gallery_emb_path.resolve()).replace("\\", "/"),
        "manifest_path": str(manifest_path.resolve()).replace("\\", "/"),
    }
    summary_path = final_output_dir / f"{args.export_prefix}_embedding_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Batch export VPRTempo qg similarity matrices for real datasets.")
    parser.add_argument("--model-path", required=True, help="Path to a trained VPRTempo .pth model")
    parser.add_argument("--data-dir", required=True, help="Base directory containing the query image folder")
    parser.add_argument("--query-dir", default=None, help="Query image folder name under data-dir")
    parser.add_argument("--gallery-dir", default=None, help="Optional gallery image folder name under data-dir for formal qg export")
    parser.add_argument("--dataset", default="custom", help="Dataset label used when generating the temporary CSV")
    parser.add_argument("--database-dirs", default="database", help="Database directory label stored in the runtime config")
    parser.add_argument("--query-places", type=int, default=None, help="Optional limit on number of query images")
    parser.add_argument("--gallery-places", type=int, default=None, help="Optional limit on number of gallery images")
    parser.add_argument("--skip", type=int, default=0, help="Number of initial query images to skip")
    parser.add_argument("--filter", type=int, default=1, help="Stride used when sub-sampling query images")
    parser.add_argument("--skip-query", type=int, default=0, help="Number of initial query images to skip in formal qg export")
    parser.add_argument("--skip-gallery", type=int, default=0, help="Number of initial gallery images to skip in formal qg export")
    parser.add_argument("--filter-query", type=int, default=1, help="Stride used when sub-sampling query images in formal qg export")
    parser.add_argument("--filter-gallery", type=int, default=1, help="Stride used when sub-sampling gallery images in formal qg export")
    parser.add_argument("--gt-tolerance", type=int, default=0, help="Ground-truth tolerance copied into VPRTempo runtime args")
    parser.add_argument("--dims", default="56,56", help="Resize dimensions, e.g. 56,56")
    parser.add_argument("--patches", type=int, default=15, help="Patch count for patch normalization")
    parser.add_argument("--export-prefix", default="vprtempo", help="Prefix used when saving the exported files")
    parser.add_argument("--output-dir", default=None, help="Optional directory to copy final qg/manifest files into")
    parser.add_argument("--manifest-dataset-name", default="custom_query", help="Dataset key written into the LSTKC manifest")
    parser.add_argument("--workers", type=int, default=0, help="Number of DataLoader workers to use during export")
    parser.add_argument("--lstkc-dataset-name", default=None, help="Optional LSTKC dataset name whose exact query/gallery ordering should be used")
    parser.add_argument("--lstkc-data-root", default=str((ROOT.parent / "AAAI2024-LSTKC" / "PRID").resolve()), help="Dataset root passed into the LSTKC dataset loader")
    parser.add_argument("--lstkc-repo-root", default=str((ROOT.parent / "AAAI2024-LSTKC").resolve()), help="Path to the AAAI2024-LSTKC repository used for split loading")
    parser.add_argument("--export-embeddings", action="store_true", help="Export normalized query/gallery embeddings and an LSTKC embedding manifest")
    args = parser.parse_args()
    args.dims = [int(x) for x in args.dims.split(",")]
    if not args.lstkc_dataset_name and not args.query_dir:
        parser.error("--query-dir is required unless --lstkc-dataset-name is provided")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.export_embeddings:
        export_embeddings_from_query_gallery(parsed)
    elif parsed.gallery_dir:
        export_qg_from_query_gallery(parsed)
    else:
        export_qg(parsed)
