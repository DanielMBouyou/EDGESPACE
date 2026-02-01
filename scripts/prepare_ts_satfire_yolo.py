import argparse
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile as tiff
from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image


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
        # Expected filename: YYYY-MM-DD_SUBDIR.tif
        date = filename.split("_")[0]
        key = (event_id, date)
        by_key.setdefault(key, {})[subdir] = name

    pairs = []
    for (event_id, date), subdirs in by_key.items():
        if "VIIRS_Day" in subdirs and "FirePred" in subdirs:
            pairs.append(
                {
                    "event_id": event_id,
                    "date": date,
                    "viirs": subdirs["VIIRS_Day"],
                    "firepred": subdirs["FirePred"],
                }
            )
    return pairs


def ensure_download(
    api: KaggleApi,
    dataset: str,
    remote_name: str,
    raw_root: Path,
    retries: int = 3,
    backoff: float = 2.0,
) -> Path | None:
    rel_parts = remote_name.split("/")[1:]  # drop dataset prefix
    local_path = raw_root.joinpath(*rel_parts)
    if local_path.exists():
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            api.dataset_download_file(dataset, remote_name, path=str(local_path.parent), quiet=False)
            return local_path
        except Exception as exc:  # noqa: BLE001
            if attempt == retries:
                print(f"Download failed after {retries} attempts: {remote_name} ({exc})")
                return None
            time.sleep(backoff * attempt)


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


def choose_fire_band(sample_masks: list[np.ndarray], threshold: float, min_frac: float, max_frac: float) -> int:
    if not sample_masks:
        return 0
    num_bands = min(mask.shape[2] for mask in sample_masks if mask.ndim == 3)
    stats = []
    for b in range(num_bands):
        fracs = []
        for mask in sample_masks:
            ch = mask[..., b]
            finite = np.isfinite(ch)
            if not finite.any():
                continue
            frac = float((ch[finite] > threshold).mean())
            fracs.append(frac)
        if fracs:
            avg = float(sum(fracs) / len(fracs))
            stats.append((avg, b))
    if not stats:
        return 0

    candidates = [s for s in stats if min_frac < s[0] < max_frac]
    if not candidates:
        candidates = [s for s in stats if s[0] > 0]
    if not candidates:
        return min(stats, key=lambda x: x[0])[1]
    return min(candidates, key=lambda x: x[0])[1]


def build_yolo_dataset(
    pairs: list[dict],
    raw_root: Path,
    out_root: Path,
    viirs_bands: list[int],
    fire_band: int,
    fire_threshold: float,
    min_area: int,
    train_split: float,
    seed: int,
) -> None:
    random.Random(seed).shuffle(pairs)
    split_idx = int(len(pairs) * train_split)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    for split in ("train", "val"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    index = []
    for i, pair in enumerate(pairs):
        split = "train" if i < split_idx else "val"
        viirs_path = raw_root.joinpath(*pair["viirs"].split("/")[1:])
        fire_path = raw_root.joinpath(*pair["firepred"].split("/")[1:])

        viirs = tiff.imread(viirs_path)
        if viirs.ndim != 3:
            raise ValueError(f"Unexpected VIIRS shape: {viirs.shape} for {viirs_path}")

        bands = []
        for b in viirs_bands:
            if b >= viirs.shape[2]:
                raise ValueError(f"VIIRS band {b} out of range for {viirs_path}")
            bands.append(scale_to_uint8(viirs[..., b]))
        img = np.stack(bands, axis=2)

        img_name = f"{pair['event_id']}_{pair['date']}_{i:05d}.jpg"
        img_path = out_root / "images" / split / img_name
        Image.fromarray(img).save(img_path, quality=92)

        fire = tiff.imread(fire_path)
        if fire.ndim != 3:
            raise ValueError(f"Unexpected FirePred shape: {fire.shape} for {fire_path}")
        use_band = fire_band if fire_band < fire.shape[2] else (fire.shape[2] - 1)
        mask = fire[..., use_band]
        mask = np.nan_to_num(mask, nan=0.0)

        if mask.shape[0] != img.shape[0] or mask.shape[1] != img.shape[1]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        binary = (mask > fire_threshold).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        yolo_lines = []
        for lbl in range(1, num_labels):
            x, y, w, h, area = stats[lbl]
            if area < min_area:
                continue
            x_center = (x + w / 2.0) / img.shape[1]
            y_center = (y + h / 2.0) / img.shape[0]
            w_norm = w / img.shape[1]
            h_norm = h / img.shape[0]
            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        label_path = out_root / "labels" / split / img_name.replace(".jpg", ".txt")
        label_path.write_text("\n".join(yolo_lines))

        index.append(
            {
                "id": i,
                "split": split,
                "event_id": pair["event_id"],
                "date": pair["date"],
                "viirs_file": pair["viirs"],
                "firepred_file": pair["firepred"],
                "image": str(img_path),
                "label": str(label_path),
            }
        )

    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {out_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: fire",
                "",
            ]
        )
    )

    (out_root / "index.json").write_text(json.dumps(index, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TS-SatFire for fast YOLO training.")
    parser.add_argument("--dataset", default="z789456sx/ts-satfire")
    parser.add_argument("--out", default="data/ts_satfire_yolo")
    parser.add_argument("--raw", default="data/raw/ts-satfire")
    parser.add_argument("--max-pairs", type=int, default=50)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--viirs-bands", default="0,1,2")
    parser.add_argument("--fire-band", type=int, default=-1)
    parser.add_argument("--fire-threshold", type=float, default=0.5)
    parser.add_argument("--min-fire-frac", type=float, default=0.001)
    parser.add_argument("--max-fire-frac", type=float, default=0.2)
    parser.add_argument("--min-area", type=int, default=20)
    args = parser.parse_args()

    out_root = Path(args.out)
    raw_root = Path(args.raw)
    out_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    viirs_bands = [int(b.strip()) for b in args.viirs_bands.split(",") if b.strip()]

    api = KaggleApi()
    api.authenticate()

    file_names = list_dataset_files(api, args.dataset)
    pairs = build_pairs(file_names)
    if not pairs:
        raise RuntimeError("No VIIRS_Day + FirePred pairs found.")

    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[: args.max_pairs]

    # Download the needed files
    ready_pairs = []
    for pair in pairs:
        viirs_path = ensure_download(api, args.dataset, pair["viirs"], raw_root)
        fire_path = ensure_download(api, args.dataset, pair["firepred"], raw_root)
        if viirs_path and fire_path:
            ready_pairs.append(pair)

    if not ready_pairs:
        raise RuntimeError("All downloads failed. Check your connection and Kaggle token.")

    pairs = ready_pairs

    # Auto-select fire band if not provided
    fire_band = args.fire_band
    if fire_band < 0:
        sample_masks = []
        for pair in pairs[: min(10, len(pairs))]:
            fire_path = raw_root.joinpath(*pair["firepred"].split("/")[1:])
            sample_masks.append(tiff.imread(fire_path))
        fire_band = choose_fire_band(sample_masks, args.fire_threshold, args.min_fire_frac, args.max_fire_frac)

    (out_root / "fire_band.txt").write_text(str(fire_band))

    build_yolo_dataset(
        pairs=pairs,
        raw_root=raw_root,
        out_root=out_root,
        viirs_bands=viirs_bands,
        fire_band=fire_band,
        fire_threshold=args.fire_threshold,
        min_area=args.min_area,
        train_split=args.train_split,
        seed=args.seed,
    )

    print(f"Prepared {len(pairs)} pairs at {out_root}")
    print(f"Fire band: {fire_band}")


if __name__ == "__main__":
    main()
