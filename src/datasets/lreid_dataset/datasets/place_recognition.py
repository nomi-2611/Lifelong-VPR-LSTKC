from __future__ import division, print_function, absolute_import
import csv
import os
import re
import os.path as osp
import numpy as np

from src.datasets.lreid_dataset.incremental_datasets import IncrementalPersonReIDSamples


VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
POSITION_KEYS = (
    "position_id",
    "place_id",
    "pid",
    "index",
    "frame_idx",
    "frame_id",
    "timestamp",
)

MSLS_DEFAULT_SPLITS = {
    "train": [
        "trondheim", "london", "boston", "melbourne", "amsterdam", "helsinki",
        "tokyo", "toronto", "saopaulo", "moscow", "zurich", "paris",
        "bangkok", "budapest", "austin", "berlin", "ottawa", "phoenix",
        "goa", "amman", "nairobi", "manila",
    ],
    "val": ["cph", "sf"],
    "test": ["miami", "athens", "buenosaires", "stockholm", "bengaluru", "kampala"],
}


def _extract_image_name(row):
    for key in ("Image_name", "Image name", "image_name", "image name", "filename", "file_name"):
        value = row.get(key)
        if value:
            return value
    return None


def _coerce_position_id(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if "." in text:
            return int(float(text))
        return int(text)
    except ValueError:
        return text


def _infer_position_id_from_name(image_name):
    stem = osp.splitext(osp.basename(str(image_name)))[0]
    matches = re.findall(r"\d+", stem)
    if matches:
        return int(matches[-1])
    return stem


def _extract_position_id(row, image_name):
    for key in POSITION_KEYS:
        value = row.get(key)
        position_id = _coerce_position_id(value)
        if position_id is not None:
            return position_id
    return _infer_position_id_from_name(image_name)


def _extract_msls_key(row):
    for key in ("key", "Key", "image_key", "Image_key", "Image name", "Image_name"):
        value = row.get(key)
        if value:
            return str(value).strip()
    return None


def _extract_msls_pid(row, image_key):
    for key in ("unique_cluster", "cluster", "place_id", "pid"):
        value = row.get(key)
        position_id = _coerce_position_id(value)
        if position_id is not None:
            return position_id
    return _infer_position_id_from_name(image_key)


def _load_csv_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _list_images_from_dir(seq_dir, csv_path=None):
    if csv_path is not None and osp.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            entries = []
            for row in reader:
                image_name = _extract_image_name(row)
                if not image_name:
                    continue
                entries.append((image_name, _extract_position_id(row, image_name)))
        return entries
    image_entries = []
    for dirpath, _, filenames in os.walk(seq_dir):
        for filename in filenames:
            if filename.lower().endswith(VALID_EXTS):
                full_path = osp.join(dirpath, filename)
                image_name = osp.relpath(full_path, seq_dir)
                image_entries.append((image_name, _infer_position_id_from_name(image_name)))
    return sorted(image_entries, key=lambda item: item[0])


def _load_pose_xyz(pose_path):
    poses = []
    with open(pose_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            values = text.split()
            if len(values) < 3:
                continue
            poses.append(tuple(float(v) for v in values[:3]))
    return poses


def _sorted_image_paths(image_dir):
    return sorted(
        [
            osp.join(image_dir, filename)
            for filename in os.listdir(image_dir)
            if filename.lower().endswith(VALID_EXTS)
        ]
    )


def _quantize_pose_xyz(xyz, step):
    pose = np.asarray(xyz, dtype=np.float32)
    return tuple(int(np.round(value / float(step))) for value in pose.tolist())


def _parse_vpr_dataset_filename_metadata(image_path):
    name = osp.basename(str(image_path))
    parts = name.split("@")
    if len(parts) < 4:
        return None
    try:
        easting = float(parts[1])
        northing = float(parts[2])
    except ValueError:
        return None
    return easting, northing


def _quantize_xy(easting, northing, step):
    return int(np.round(float(easting) / float(step))), int(np.round(float(northing) / float(step)))


class IncrementalSamples4PlaceBase(IncrementalPersonReIDSamples):
    dataset_name = "place_base"
    dataset_dir = ""
    train_dirs = []
    query_dirs = []
    gallery_dirs = []
    csv_map = {}
    group_by_scene = False
    auto_discover = False

    def __init__(self, datasets_root, relabel=True, combineall=False):
        self.relabel = relabel
        self.combineall = combineall
        self.root = osp.join(datasets_root, self.dataset_dir)
        self._place_to_pid = {}

        train_dirs, query_dirs, gallery_dirs = self._resolve_dirs()
        train = self._build_samples(train_dirs, split="train")
        query = self._build_samples(query_dirs, split="query")
        gallery = self._build_samples(gallery_dirs, split="gallery")
        self.train, self.query, self.gallery = train, query, gallery
        self._show_info(train, query, gallery)

    def _resolve_dirs(self):
        if self.auto_discover:
            dir_names = [d for d in os.listdir(self.root) if osp.isdir(osp.join(self.root, d))]
            train_dirs = sorted([d for d in dir_names if "Easy_image_left" in d])
            query_dirs = sorted([d for d in dir_names if "Hard_image_left" in d])
            gallery_dirs = train_dirs
            return train_dirs, query_dirs, gallery_dirs
        return list(self.train_dirs), list(self.query_dirs), list(self.gallery_dirs)

    def _scene_key(self, seq_name):
        if not self.group_by_scene:
            return "shared_route"
        return re.sub(r"_(Easy|Hard)_image_left$", "", seq_name)

    def _pid_for(self, scene_key, position_id):
        key = (scene_key, position_id)
        if key not in self._place_to_pid:
            self._place_to_pid[key] = len(self._place_to_pid)
        return self._place_to_pid[key]

    def _build_samples(self, seq_dirs, split):
        samples = []
        cam_base = {"train": 0, "query": 100, "gallery": 200}.get(split, 0)
        for seq_idx, seq_name in enumerate(seq_dirs):
            seq_dir = osp.join(self.root, seq_name)
            csv_name = self.csv_map.get(seq_name)
            csv_path = osp.join(self.root, csv_name) if csv_name else None
            image_entries = _list_images_from_dir(seq_dir, csv_path=csv_path)
            scene_key = self._scene_key(seq_name)
            camid = cam_base + seq_idx
            for image_name, position_id in image_entries:
                fpath = osp.join(seq_dir, image_name)
                pid = self._pid_for(scene_key, position_id)
                samples.append((fpath, pid, camid, self.dataset_name))
        return samples


class IncrementalSamples4nordlandPlace(IncrementalSamples4PlaceBase):
    dataset_name = "nordland_place"
    dataset_dir = "nordland"
    train_dirs = ["spring", "fall"]
    query_dirs = ["summer"]
    gallery_dirs = ["spring", "fall"]
    csv_map = {
        "spring": "nordland-spring.csv",
        "fall": "nordland-fall.csv",
        "summer": "nordland-summer.csv",
        "winter": "nordland-winter.csv",
    }
    group_by_scene = False


class IncrementalSamples4robotcarPlace(IncrementalSamples4PlaceBase):
    dataset_name = "robotcar_place"
    dataset_dir = "robotcar"
    train_dirs = ["sun", "rain"]
    query_dirs = ["dusk"]
    gallery_dirs = ["sun", "rain"]
    csv_map = {
        "sun": "orc-sun.csv",
        "rain": "orc-rain.csv",
        "dusk": "orc-dusk.csv",
    }
    group_by_scene = False


class IncrementalSamples4tartPlace(IncrementalSamples4PlaceBase):
    dataset_name = "tart_place"
    dataset_dir = "tart"
    auto_discover = True
    group_by_scene = True
    easy_pose_quantization_xy = 0.25
    hard_match_threshold_xy = 0.25

    def __init__(self, datasets_root, relabel=True, combineall=False):
        self._tart_pose_cache = {}
        self._tart_scene_reference_cache = {}
        super(IncrementalSamples4tartPlace, self).__init__(datasets_root, relabel=relabel, combineall=combineall)

    def _iter_tart_pose_entries(self, seq_name):
        if seq_name in self._tart_pose_cache:
            return self._tart_pose_cache[seq_name]

        seq_dir = osp.join(self.root, seq_name)
        entries = []
        for dirpath, _, filenames in os.walk(seq_dir):
            if 'pose_left.txt' not in filenames:
                continue
            pose_path = osp.join(dirpath, 'pose_left.txt')
            route_dir = dirpath
            image_dir = osp.join(route_dir, 'image_left')
            if osp.isdir(image_dir):
                image_paths = _sorted_image_paths(image_dir)
            else:
                image_paths = _sorted_image_paths(seq_dir)
            poses = _load_pose_xyz(pose_path)
            if len(image_paths) != len(poses):
                continue

            route_key = osp.relpath(route_dir, seq_dir)
            for image_path, pose_xyz in zip(image_paths, poses):
                entries.append((image_path, route_key, pose_xyz))

        self._tart_pose_cache[seq_name] = entries
        return entries

    def _pose_xy(self, pose_xyz):
        pose = np.asarray(pose_xyz, dtype=np.float32)
        return pose[:2]

    def _scene_reference_for_easy(self, easy_seq_name):
        if easy_seq_name in self._tart_scene_reference_cache:
            return self._tart_scene_reference_cache[easy_seq_name]

        references = {}
        grid = {}
        for image_path, _, pose_xyz in self._iter_tart_pose_entries(easy_seq_name):
            cell = _quantize_pose_xyz(self._pose_xy(pose_xyz), self.easy_pose_quantization_xy)
            if cell not in references:
                references[cell] = {
                    "xy": self._pose_xy(pose_xyz),
                    "pid": None,
                }
                grid.setdefault(cell, []).append(cell)

        payload = {
            "references": references,
            "grid": grid,
        }
        self._tart_scene_reference_cache[easy_seq_name] = payload
        return payload

    def _assign_hard_pid(self, scene_key, easy_seq_name, pose_xyz):
        payload = self._scene_reference_for_easy(easy_seq_name)
        references = payload["references"]
        if not references:
            return None
        query_xy = self._pose_xy(pose_xyz)
        query_cell = _quantize_pose_xyz(query_xy, self.easy_pose_quantization_xy)
        search_radius = max(1, int(np.ceil(self.hard_match_threshold_xy / float(self.easy_pose_quantization_xy))))
        candidates = []
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                neighbor = (query_cell[0] + dx, query_cell[1] + dy)
                if neighbor in references:
                    candidates.append(neighbor)
        if not candidates:
            return None

        best_cell = None
        best_distance = None
        for cell in candidates:
            item = references[cell]
            distance = float(np.linalg.norm(item["xy"] - query_xy))
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_cell = cell
        if best_cell is None or best_distance is None or best_distance > float(self.hard_match_threshold_xy):
            return None
        return self._pid_for(scene_key, best_cell)

    def _build_samples(self, seq_dirs, split):
        samples = []
        cam_base = {"train": 0, "query": 100, "gallery": 200}.get(split, 0)
        for seq_idx, seq_name in enumerate(seq_dirs):
            scene_key = self._scene_key(seq_name)
            camid = cam_base + seq_idx
            for image_path, route_key, pose_xyz in self._iter_tart_pose_entries(seq_name):
                easy_seq_name = scene_key + '_Easy_image_left'
                if 'Easy_image_left' in seq_name:
                    position_id = _quantize_pose_xyz(self._pose_xy(pose_xyz), self.easy_pose_quantization_xy)
                    pid = self._pid_for(scene_key, position_id)
                else:
                    pid = self._assign_hard_pid(scene_key, easy_seq_name, pose_xyz)
                    if pid is None:
                        continue
                samples.append((image_path, pid, camid, self.dataset_name))
        return samples


class IncrementalSamples4mslsPlace(IncrementalPersonReIDSamples):
    dataset_name = "msls_place"
    dataset_dir = "msls"
    train_split = "train"
    eval_split = "val"

    def __init__(self, datasets_root, relabel=True, combineall=False):
        self.relabel = relabel
        self.combineall = combineall
        self.root = osp.join(datasets_root, self.dataset_dir)
        self._place_to_pid = {}
        self._cities = self._discover_cities()
        train = self._build_train_samples()
        query, gallery = self._build_eval_samples()
        self.train, self.query, self.gallery = train, query, gallery
        self._show_info(train, query, gallery)

    def _discover_cities(self):
        discovered = {"train": [], "val": [], "test": []}
        for split_name, cities in MSLS_DEFAULT_SPLITS.items():
            base_dir = "test" if split_name == "test" else "train_val"
            split_cities = []
            for city in cities:
                city_dir = osp.join(self.root, base_dir, city)
                if osp.isdir(city_dir):
                    split_cities.append(city)
            discovered[split_name] = split_cities
        return discovered

    def _pid_for(self, city, position_id):
        key = (city, position_id)
        if key not in self._place_to_pid:
            self._place_to_pid[key] = len(self._place_to_pid)
        return self._place_to_pid[key]

    def _city_base_dir(self, city):
        return "test" if city in self._cities["test"] else "train_val"

    def _subset_dir(self, city, subset):
        return osp.join(self.root, self._city_base_dir(city), city, subset)

    def _read_subset_samples(self, city, subset, camid, pid_lookup_mode="register"):
        subset_dir = self._subset_dir(city, subset)
        postprocessed_path = osp.join(subset_dir, "postprocessed.csv")
        raw_path = osp.join(subset_dir, "raw.csv")
        image_dir = osp.join(subset_dir, "images")
        if not osp.exists(postprocessed_path) or not osp.isdir(image_dir):
            return []

        rows = _load_csv_rows(postprocessed_path)
        raw_keys = None
        if osp.exists(raw_path):
            raw_rows = _load_csv_rows(raw_path)
            raw_keys = {key for key in (_extract_msls_key(row) for row in raw_rows) if key}

        samples = []
        for row in rows:
            image_key = _extract_msls_key(row)
            if not image_key:
                continue
            if raw_keys is not None and image_key not in raw_keys:
                continue
            image_path = osp.join(image_dir, image_key + ".jpg")
            if not osp.exists(image_path):
                continue
            position_id = _extract_msls_pid(row, image_key)
            if pid_lookup_mode == "register":
                pid = self._pid_for(city, position_id)
            else:
                pid = self._place_to_pid.get((city, position_id))
                if pid is None:
                    continue
            samples.append((image_path, pid, camid, self.dataset_name))
        return samples

    def _build_train_samples(self):
        train_cities = list(self._cities["train"])
        if not train_cities:
            train_cities = list(self._cities["val"])
        samples = []
        for city_idx, city in enumerate(train_cities):
            database_camid = city_idx * 2
            query_camid = city_idx * 2 + 1
            samples.extend(self._read_subset_samples(city, "database", database_camid, pid_lookup_mode="register"))
            samples.extend(self._read_subset_samples(city, "query", query_camid, pid_lookup_mode="register"))
        return samples

    def _build_eval_samples(self):
        eval_cities = list(self._cities["val"])
        if not eval_cities:
            eval_cities = list(self._cities["train"])
        query_samples = []
        gallery_samples = []
        for city_idx, city in enumerate(eval_cities):
            query_camid = 100 + city_idx * 2
            gallery_camid = 200 + city_idx * 2
            gallery_samples.extend(
                self._read_subset_samples(city, "database", gallery_camid, pid_lookup_mode="register")
            )
            query_samples.extend(
                self._read_subset_samples(city, "query", query_camid, pid_lookup_mode="known_only")
            )
        return query_samples, gallery_samples


class IncrementalSamples4pitts30kPlace(IncrementalPersonReIDSamples):
    dataset_name = "pitts30k_place"
    dataset_dir = "pitts30k"
    train_split = "train"
    eval_split = "test"
    pid_grid_size_m = 10.0

    def __init__(self, datasets_root, relabel=True, combineall=False):
        self.relabel = relabel
        self.combineall = combineall
        self.root = osp.join(datasets_root, self.dataset_dir)
        self._place_to_pid = {}
        train = self._build_split_samples(self.train_split, include_query=True, cam_base=0, register=True)
        gallery = self._build_subset_samples(self.eval_split, "database", camid=200, register=True)
        query = self._build_subset_samples(self.eval_split, "queries", camid=100, register=False)
        self.train, self.query, self.gallery = train, query, gallery
        self._show_info(train, query, gallery)

    def _pid_for(self, split_name, xy_key, register=True):
        key = (split_name, xy_key)
        if key not in self._place_to_pid:
            if not register:
                return None
            self._place_to_pid[key] = len(self._place_to_pid)
        return self._place_to_pid[key]

    def _subset_dir(self, split_name, subset):
        return osp.join(self.root, "images", split_name, subset)

    def _build_subset_samples(self, split_name, subset, camid, register=True):
        subset_dir = self._subset_dir(split_name, subset)
        if not osp.isdir(subset_dir):
            return []
        samples = []
        for image_path in _sorted_image_paths(subset_dir):
            metadata = _parse_vpr_dataset_filename_metadata(image_path)
            if metadata is None:
                continue
            easting, northing = metadata
            xy_key = _quantize_xy(easting, northing, self.pid_grid_size_m)
            pid = self._pid_for(split_name, xy_key, register=register)
            if pid is None:
                continue
            samples.append((image_path, pid, camid, self.dataset_name))
        return samples

    def _build_split_samples(self, split_name, include_query, cam_base, register):
        samples = []
        samples.extend(self._build_subset_samples(split_name, "database", cam_base, register=register))
        if include_query:
            samples.extend(self._build_subset_samples(split_name, "queries", cam_base + 1, register=register))
        return samples
