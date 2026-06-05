#!/usr/bin/env python3
"""Convert DroneVehice RGB-IR XML annotations to cropped YOLO format."""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


CLASS_NAMES = ["car", "truck", "bus", "van", "feright_car"]
CLASS_MAP = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "van": 3,
    "feright_car": 4,
    "feright car": 4,
    "feright": 4,
    "freight_car": 4,
    "freight car": 4,
    "freight": 4,
}
SPLITS = {
    "train": ("trainimg", "trainimgr", "trainlabelr"),
    "val": ("valimg", "valimgr", "vallabelr"),
    "test": ("testimg", "testimgr", "testlabelr"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DroneVehice to YOLO RGB-IR dataset.")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/home/user/4T_Storage/SJY/Infrared/datasets/Detection/DroneVehice"),
        help="Original DroneVehice dataset root.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("/home/user/4T_Storage/SJY/Infrared/datasets/Detection/DroneVehice_yolo_640x512"),
        help="Output YOLO dataset root.",
    )
    parser.add_argument("--width", type=int, default=640, help="Output image width.")
    parser.add_argument("--height", type=int, default=512, help="Output image height.")
    parser.add_argument("--overwrite", action="store_true", help="Remove output directory before conversion.")
    return parser.parse_args()


def crop_box_for_image(image: np.ndarray, out_w: int, out_h: int) -> tuple[float, float, float, float]:
    """Return the centered crop box that removes the DroneVehicle white border."""
    h, w = image.shape[:2]
    if w >= out_w and h >= out_h:
        left = (w - out_w) / 2.0
        top = (h - out_h) / 2.0
        return left, top, left + out_w, top + out_h
    return 0.0, 0.0, float(w), float(h)


def crop_resize(image: np.ndarray, crop: tuple[float, float, float, float], out_w: int, out_h: int) -> np.ndarray:
    """Crop image and resize to target size."""
    left, top, right, bottom = crop
    h, w = image.shape[:2]
    x1, y1 = max(0, int(round(left))), max(0, int(round(top)))
    x2, y2 = min(w, int(round(right))), min(h, int(round(bottom)))
    cropped = image[y1:y2, x1:x2]
    if cropped.shape[1] != out_w or cropped.shape[0] != out_h:
        cropped = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return cropped


def polygon_points(obj: ET.Element) -> Optional[np.ndarray]:
    """Extract polygon or bndbox points from an annotation object."""
    polygon = obj.find("polygon")
    if polygon is not None:
        pts = []
        for i in range(1, 5):
            x = polygon.findtext(f"x{i}")
            y = polygon.findtext(f"y{i}")
            if x is None or y is None:
                return None
            pts.append((float(x), float(y)))
        return np.array(pts, dtype=np.float32)

    bndbox = obj.find("bndbox")
    if bndbox is not None:
        vals = [bndbox.findtext(k) for k in ("xmin", "ymin", "xmax", "ymax")]
        if any(v is None for v in vals):
            return None
        xmin, ymin, xmax, ymax = (float(v) for v in vals)
        return np.array([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)], dtype=np.float32)
    return None


def convert_xml_label(
    xml_file: Path,
    crop: tuple[float, float, float, float],
    out_w: int,
    out_h: int,
) -> tuple[list[str], int, int]:
    """Convert one IR XML annotation to YOLO normalized labels."""
    left, top, right, bottom = crop
    crop_w, crop_h = right - left, bottom - top
    sx, sy = out_w / crop_w, out_h / crop_h
    labels: list[str] = []
    skipped_unknown = 0
    skipped_box = 0

    if not xml_file.exists():
        return labels, skipped_unknown, skipped_box

    root = ET.parse(xml_file).getroot()
    for obj in root.findall(".//object"):
        cls_name = (obj.findtext("name") or "").strip().lower().replace("-", "_")
        cls_id = CLASS_MAP.get(cls_name)
        if cls_id is None:
            skipped_unknown += 1
            continue

        pts = polygon_points(obj)
        if pts is None:
            skipped_box += 1
            continue

        pts[:, 0] = (pts[:, 0] - left) * sx
        pts[:, 1] = (pts[:, 1] - top) * sy
        x1 = float(np.clip(pts[:, 0].min(), 0, out_w))
        y1 = float(np.clip(pts[:, 1].min(), 0, out_h))
        x2 = float(np.clip(pts[:, 0].max(), 0, out_w))
        y2 = float(np.clip(pts[:, 1].max(), 0, out_h))
        bw, bh = x2 - x1, y2 - y1
        if bw <= 1.0 or bh <= 1.0:
            skipped_box += 1
            continue

        xc = (x1 + x2) / 2.0 / out_w
        yc = (y1 + y2) / 2.0 / out_h
        labels.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw / out_w:.6f} {bh / out_h:.6f}")

    return labels, skipped_unknown, skipped_box


def convert_split(src: Path, dst: Path, split: str, out_w: int, out_h: int) -> dict[str, int]:
    """Convert one split."""
    rgb_dir_name, ir_dir_name, ir_label_dir_name = SPLITS[split]
    rgb_src = src / split / rgb_dir_name
    ir_src = src / split / ir_dir_name
    label_src = src / split / ir_label_dir_name
    rgb_dst = dst / "images" / split
    ir_dst = dst / "ir_images" / split
    label_dst = dst / "labels" / split
    for d in (rgb_dst, ir_dst, label_dst):
        d.mkdir(parents=True, exist_ok=True)

    stats = {
        "images": 0,
        "missing_rgb": 0,
        "missing_ir": 0,
        "missing_label": 0,
        "objects": 0,
        "skipped_unknown": 0,
        "skipped_box": 0,
    }

    for ir_file in sorted(ir_src.glob("*.jpg")):
        stem = ir_file.stem
        rgb_file = rgb_src / f"{stem}.jpg"
        xml_file = label_src / f"{stem}.xml"
        if not rgb_file.exists():
            stats["missing_rgb"] += 1
            continue
        if not ir_file.exists():
            stats["missing_ir"] += 1
            continue
        if not xml_file.exists():
            stats["missing_label"] += 1

        ir_img = cv2.imread(str(ir_file), cv2.IMREAD_COLOR)
        rgb_img = cv2.imread(str(rgb_file), cv2.IMREAD_COLOR)
        if ir_img is None or rgb_img is None:
            stats["missing_ir" if ir_img is None else "missing_rgb"] += 1
            continue

        crop = crop_box_for_image(ir_img, out_w, out_h)
        ir_out = crop_resize(ir_img, crop, out_w, out_h)
        rgb_out = crop_resize(rgb_img, crop, out_w, out_h)

        rgb_name = f"{stem}.jpg"
        ir_name = f"ir_{stem}.jpg"
        cv2.imwrite(str(rgb_dst / rgb_name), rgb_out)
        cv2.imwrite(str(ir_dst / ir_name), ir_out)

        labels, skipped_unknown, skipped_box = convert_xml_label(xml_file, crop, out_w, out_h)
        stats["objects"] += len(labels)
        stats["skipped_unknown"] += skipped_unknown
        stats["skipped_box"] += skipped_box

        # Write both names so standard YOLO RGB and paired RGB-IR loaders can use the same label set.
        label_text = "\n".join(labels) + ("\n" if labels else "")
        (label_dst / f"{stem}.txt").write_text(label_text, encoding="utf-8")
        (label_dst / f"ir_{stem}.txt").write_text(label_text, encoding="utf-8")
        stats["images"] += 1

    return stats


def write_yaml(dst: Path) -> None:
    """Write data.yaml for paired RGB-IR YOLO training."""
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    text = f"""path: {dst}
train: ir_images/train
val: ir_images/val
test: ir_images/test

train_rgb: images/train
val_rgb: images/val
test_rgb: images/test

nc: {len(CLASS_NAMES)}
names:
{names}
channels: 3
"""
    (dst / "data.yaml").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.overwrite and args.dst.exists():
        shutil.rmtree(args.dst)
    args.dst.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for split in SPLITS:
        stats = convert_split(args.src, args.dst, split, args.width, args.height)
        all_stats[split] = stats
        print(f"{split}: {stats}")
    write_yaml(args.dst)
    print(f"data.yaml: {args.dst / 'data.yaml'}")


if __name__ == "__main__":
    main()
