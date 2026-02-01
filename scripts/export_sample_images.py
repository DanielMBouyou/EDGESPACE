import argparse
import os
import random
from pathlib import Path

import numpy as np
import tifffile as tiff
from PIL import Image

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except Exception:  # noqa: BLE001
    KaggleApi = None


def scale_to_uint8(channel: np.ndarray) -> np.ndarray:
    finite = np.isfinite(channel)
    if not finite.any():
        return np.zeros(channel.shape, dtype=np.uint8)
    minv = float(channel[finite].min())
    maxv = float(channel[finite].max())
    if maxv <= minv:
        return np.zeros(channel.shape, dtype=np.uint8)
    filled = np.nan_to_num(channel, nan=minv)
    norm = (filled - minv) / (maxv - minv)
    norm = np.clip(norm, 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def list_dataset_files(api: KaggleApi, dataset: str) -> list[str]:
    files: list[str] = []
    page_token = None
    while True:
        resp = api.dataset_list_files(dataset, page_token=page_token, page_size=200)
        files.extend([f.name for f in resp.dataset_files])
        page_token = resp.nextPageToken
        if not page_token:
            break
    return files


def build_pairs(file_names: list[str]) -> list[dict]:
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for name in file_names:
        parts = name.split("/")
        if len(parts) < 4:
            continue
        _, event_id, subdir, filename = parts[0], parts[1], parts[2], parts[3]
        if not filename.endswith(".tif"):
            continue
        date = filename.split("_")[0]
        key = (event_id, date)
        by_key.setdefault(key, {})[subdir] = name

    pairs = []
    for (event_id, date), subdirs in by_key.items():
        if "VIIRS_Day" in subdirs:
            pairs.append({"event_id": event_id, "date": date, "viirs": subdirs["VIIRS_Day"]})
    return pairs


def ensure_download(api: KaggleApi, dataset: str, remote_name: str, raw_root: Path) -> Path:
    rel_parts = remote_name.split("/")[1:]
    local_path = raw_root.joinpath(*rel_parts)
    if local_path.exists():
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    api.dataset_download_file(dataset, remote_name, path=str(local_path.parent), quiet=False)
    return local_path


def load_viirs_rgb(path: Path, bands: list[int]) -> np.ndarray:
    arr = tiff.imread(path)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected VIIRS shape: {arr.shape} for {path}")
    if max(bands) >= arr.shape[2]:
        raise ValueError(f"VIIRS bands {bands} out of range for {path}")
    stacked = [scale_to_uint8(arr[..., b]) for b in bands]
    return np.stack(stacked, axis=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a few TS-SatFire samples as JPGs.")
    parser.add_argument("--dataset", default="z789456sx/ts-satfire")
    parser.add_argument("--raw", default="data/raw/ts-satfire")
    parser.add_argument("--out", default="sample_images")
    parser.add_argument("--num", type=int, default=4)
    parser.add_argument("--bands", default="0,1,2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_root = Path(args.raw)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    bands = [int(b.strip()) for b in args.bands.split(",") if b.strip()]

    viirs_files = list(raw_root.rglob("*_VIIRS_Day.tif"))
    if len(viirs_files) < args.num:
        if KaggleApi is None:
            raise RuntimeError("Kaggle API not available. Install 'kaggle' or download data first.")
        api = KaggleApi()
        api.authenticate()
        file_names = list_dataset_files(api, args.dataset)
        pairs = build_pairs(file_names)
        random.Random(args.seed).shuffle(pairs)
        for pair in pairs:
            viirs_path = ensure_download(api, args.dataset, pair["viirs"], raw_root)
            viirs_files.append(viirs_path)
            if len(viirs_files) >= args.num:
                break

    random.Random(args.seed).shuffle(viirs_files)
    viirs_files = viirs_files[: args.num]

    for idx, viirs_path in enumerate(viirs_files, start=1):
        rgb = load_viirs_rgb(viirs_path, bands)
        name = viirs_path.stem.replace("_VIIRS_Day", "")
        out_path = out_root / f"sample_{idx:02d}_{name}.jpg"
        Image.fromarray(rgb).save(out_path, quality=92)

    print(f"Saved {len(viirs_files)} samples to {out_root}")


if __name__ == "__main__":
    main()
