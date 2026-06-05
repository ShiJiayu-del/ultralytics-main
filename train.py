#!/usr/bin/env python3
"""Simple training entrypoint for Ultralytics YOLO."""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO model")
    parser.add_argument("--model", type=str, required=True, help="model yaml or pt path")
    parser.add_argument("--data", type=str, required=True, help="dataset yaml path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--optimizer", type=str, default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--weights", type=str, default="", help="pretrained checkpoint path")
    parser.add_argument("--device", type=str, default="", help="CUDA device(s), e.g. '0' or '2,3', or 'cpu'")
    parser.add_argument("--workers", type=int, default=8, help="number of dataloader workers")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True, help="enable AMP mixed precision")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="disable AMP mixed precision")
    parser.add_argument("--rect", action="store_true", help="use rectangular training batches")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        optimizer=args.optimizer,
        lr0=args.lr0,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        pretrained=args.weights if args.weights else True,
        device=args.device,
        workers=args.workers,
        amp=args.amp,
        rect=args.rect,
    )


if __name__ == "__main__":
    main()
