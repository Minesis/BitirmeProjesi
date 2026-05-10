"""
main.py
--------
Unified entry-point for the Retail Analytics System.

Usage:
    python main.py detect          # Run multi-camera (from config/config.yaml)
    python main.py detect --source 0 --show         # Quick single-camera test
    python main.py detect --source data/store.mp4   # Run on video file
    python main.py api             # Start FastAPI backend
    python main.py train           # Train age/gender model
    python main.py demo            # Insert synthetic demo data for testing
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
#  Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging() -> None:
    log_cfg = cfg.get("logging", default={})
    log_file = log_cfg.get("log_file", "logs/retail_analytics.log")
    level = log_cfg.get("level", "INFO")
    rotation = log_cfg.get("rotation", "10 MB")
    os.makedirs(Path(log_file).parent, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    logger.add(log_file, level=level, rotation=rotation, compression="zip")


# ─────────────────────────────────────────────────────────────────────────────
#  Sub-commands
# ─────────────────────────────────────────────────────────────────────────────

def run_detect(args: argparse.Namespace) -> None:
    """Run multi-camera shelf analytics pipeline."""
    from app.vision.camera_manager import CameraManager

    # If --source is provided, run a single temporary camera config for quick tests.
    if args.source is not None:
        camera_id = args.camera_id or "cam_1"
        shelf_name = args.shelf_name or "Shelf"
        source: int | str = args.source
        try:
            source = int(source)
        except (ValueError, TypeError):
            pass
        camera_configs = [{"id": camera_id, "shelf_name": shelf_name, "source": source}]
        mgr = CameraManager(show=args.show, camera_configs=camera_configs)
    else:
        mgr = CameraManager(show=args.show, camera_id_filter=args.camera_id)

    logger.info("Multi-camera analytics started. Press Ctrl+C to stop.")
    mgr.run_forever(restart_dead_workers=not args.no_restart)


def run_api(args: argparse.Namespace) -> None:
    """Start the FastAPI backend."""
    import uvicorn
    from app.api.app import create_app

    app = create_app()
    host = cfg.get("api.host", default="0.0.0.0")
    port = cfg.get("api.port", default=8000)
    logger.info(f"Starting API on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


def run_train(args: argparse.Namespace) -> None:
    """Train the age/gender model."""
    from app.model.train import train

    train(
        data_dir=args.data_dir or "data/UTKFace",
        save_dir=args.save_dir or "models",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        input_size=args.input_size,
        device_str=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
    )


def run_demo(args: argparse.Namespace) -> None:
    """Insert synthetic data so the dashboard shows something before real data is collected."""
    import random
    from datetime import datetime, timedelta
    from app.database.repository import AnalyticsRepository

    db_path = cfg.get("database.path", default="data/analytics.db")
    os.makedirs(Path(db_path).parent, exist_ok=True)
    repo = AnalyticsRepository(db_path)

    cameras_cfg = list(cfg.get("cameras", default=[]) or [])
    if not cameras_cfg:
        cameras_cfg = [{"id": "cam_1", "shelf_name": "Demo Shelf", "source": 0}]

    genders = ["Male", "Female"]
    age_groups = list(cfg.get("age_gender.age_labels", default=[])) or [
        "0-12",
        "13-17",
        "18-24",
        "25-34",
        "35-44",
        "45-54",
        "55-64",
        "65-74",
        "75-84",
        "85+",
    ]

    n = args.count
    logger.info(f"Inserting {n} demo visits …")

    now = datetime.utcnow()

    for i in range(n):
        cam = random.choice(cameras_cfg)
        camera_id = str(cam.get("id", "cam_1"))
        shelf_name = str(cam.get("shelf_name", camera_id))
        gender = random.choice(genders)
        age_group = random.choice(age_groups)
        duration = float(random.uniform(10, 300))
        enter_time = now - timedelta(seconds=random.uniform(0, 24 * 3600))
        exit_time = enter_time + timedelta(seconds=duration)

        visit_id = repo.open_visit(
            camera_id=camera_id,
            shelf_name=shelf_name,
            visitor_id=i + 1,
            gender=gender,
            age_group=age_group,
            enter_time=enter_time,
        )
        repo.close_visit(visit_id, exit_time=exit_time)

    logger.info(f"Demo data inserted into {db_path}. Launch the dashboard to visualise.")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="retail_analytics",
        description="Retail Analytics System",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # detect
    det = sub.add_parser("detect", help="Run live detection pipeline")
    det.add_argument(
        "--source",
        default=None,
        help="Camera index, RTSP URL, or video file path. If omitted, uses cameras in config/config.yaml.",
    )
    det.add_argument("--show", action="store_true", help="Show OpenCV windows (press 'q' to quit).")
    det.add_argument(
        "--camera_id",
        default=None,
        help="Filter to a single configured camera id (or set id when using --source).",
    )
    det.add_argument(
        "--shelf_name",
        default=None,
        help="Shelf name when using --source (ignored when reading config).",
    )
    det.add_argument("--no_restart", action="store_true", help="Disable worker auto-restart watchdog.")

    # api
    sub.add_parser("api", help="Start FastAPI backend")

    # train
    tr = sub.add_parser("train", help="Train age/gender CNN model")
    tr.add_argument("--data_dir", default="data/UTKFace")
    tr.add_argument("--save_dir", default="models")
    tr.add_argument("--epochs", type=int, default=20)
    tr.add_argument("--batch_size", type=int, default=64)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--input_size", type=int, default=96)
    tr.add_argument("--device", default="cpu")
    tr.add_argument("--num_workers", type=int, default=0)
    tr.add_argument("--seed", type=int, default=42)

    # demo
    dm = sub.add_parser("demo", help="Insert synthetic demo data")
    dm.add_argument("--count", type=int, default=200, help="Number of events to generate")

    return p


if __name__ == "__main__":
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "detect": run_detect,
        "api": run_api,
        "train": run_train,
        "demo": run_demo,
    }
    dispatch[args.command](args)
