"""
Command Line Interface for the offline face clustering tool and web gallery server.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tabulate import tabulate

from app.db import Database
from app.pipeline import run_pipeline


def print_summary(stats: dict) -> None:
    """Pretty-print the scan and clustering summary."""
    print("\n" + "=" * 55)
    print("           FACE CLUSTERING SUMMARY")
    print("=" * 55)
    print(f"  Images Scanned:       {stats['images_scanned']}")
    print(f"  Faces Detected:       {stats['faces_detected']}")
    print(f"  Clusters / People:    {stats['num_clusters']}")
    if stats.get("unclustered_faces", 0) > 0:
        print(f"  Unclustered (Noise):  {stats['unclustered_faces']}")
    print("-" * 55)

    if stats.get("clusters"):
        table_data = []
        for c in stats["clusters"]:
            table_data.append([
                f"Cluster {c['cluster_id']}",
                c["faces_count"],
                c["images_count"]
            ])
        print(tabulate(
            table_data,
            headers=["Person / Cluster ID", "Faces Count", "Images Count"],
            tablefmt="simple"
        ))
    else:
        print("  No face clusters formed.")
    print("=" * 55 + "\n")


def inspect_cluster_cmd(db_path: str, cluster_id: int) -> None:
    """Inspect a cluster and print associated image paths and face details."""
    db = Database(db_path)
    res = db.inspect_cluster(cluster_id)

    if not res["faces"]:
        print(f"No faces found for Cluster ID: {cluster_id}")
        return

    print("\n" + "=" * 60)
    print(f"              CLUSTER {cluster_id} DETAILS")
    print("=" * 60)
    print(f"  Total Faces:  {res['total_faces']}")
    print(f"  Total Images: {res['total_images']}")
    print("-" * 60)
    print("  Associated Image Paths:")
    for path in res["image_paths"]:
        print(f"    • {path}")

    print("-" * 60)
    print("  Face Bounding Boxes:")
    for face in res["faces"]:
        bbox = face["bbox"]
        print(f"    [Face #{face['face_id']}] Conf: {face['confidence']:.2f} | BBox: ({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}) -> {Path(face['file_path']).name}")
    print("=" * 60 + "\n")


def serve_cmd(
    media_dir: str | None = None,
    db_path: str = "face_clusters.db",
    host: str = "127.0.0.1",
    port: int = 8000,
    rescan: bool = False,
) -> None:
    """Run the web media gallery server."""
    import uvicorn
    from app.server import create_app

    if media_dir and (rescan or not Path(db_path).exists()):
        print(f"Indexing media directory: {media_dir} ...")
        stats = run_pipeline(input_dir=media_dir, db_path=db_path)
        print_summary(stats)

    app = create_app(db_path=db_path)
    print(f"\n=======================================================")
    print(f"  ✨ Kimera Media Gallery running at: http://{host}:{port}")
    print(f"  Database: {db_path}")
    print(f"=======================================================\n")
    uvicorn.run(app, host=host, port=port, log_level="info")


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app",
        description="Kimera: Offline Media Gallery & Face Clustering Tool"
    )

    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommands")

    # Serve subcommand (Web Gallery)
    serve_parser = subparsers.add_parser("serve", help="Launch the local media gallery web UI")
    serve_parser.add_argument("path", nargs="?", type=str, default=None, help="Directory path to auto-index")
    serve_parser.add_argument("--db", type=str, default="face_clusters.db", help="SQLite database path")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    serve_parser.add_argument("--rescan", action="store_true", help="Force re-scan before launching")

    # Scan subcommand
    scan_parser = subparsers.add_parser("scan", help="Scan a directory, detect faces, and cluster them")
    scan_parser.add_argument("path", type=str, help="Directory path containing photos")
    scan_parser.add_argument("--db", type=str, default="face_clusters.db", help="SQLite database path (default: face_clusters.db)")
    scan_parser.add_argument("--threshold", type=float, default=0.65, help="Cosine distance threshold for clustering (default: 0.65)")
    scan_parser.add_argument("--conf", type=float, default=0.5, help="Face detection confidence threshold (default: 0.5)")
    scan_parser.add_argument("--algo", type=str, default="dbscan", choices=["dbscan", "agglomerative"], help="Clustering algorithm")
    scan_parser.add_argument("--export", type=str, default=None, help="Directory to export cropped face cutouts grouped by person")
    scan_parser.add_argument("--scene-thresh", type=float, default=0.35, help="Video scene cut threshold (0.0 - 1.0, default: 0.35)")
    scan_parser.add_argument("--min-interval", type=float, default=30.0, help="Minimum seconds between video keyframes (default: 30.0)")
    scan_parser.add_argument("--max-interval", type=float, default=90.0, help="Maximum seconds between video keyframe samples (default: 90.0)")

    # Inspect subcommand
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a specific person/cluster")
    inspect_parser.add_argument("cluster_id", type=int, help="Cluster ID to inspect")
    inspect_parser.add_argument("--db", type=str, default="face_clusters.db", help="SQLite database path (default: face_clusters.db)")

    if args is None:
        args = sys.argv[1:]

    # If first argument is a valid directory or path not matching subcommands, treat as scan command
    if args and args[0] not in ("scan", "inspect", "serve", "-h", "--help"):
        args = ["scan"] + args

    parsed = parser.parse_args(args)

    if parsed.subcommand == "serve":
        serve_cmd(
            media_dir=parsed.path,
            db_path=parsed.db,
            host=parsed.host,
            port=parsed.port,
            rescan=parsed.rescan,
        )
        return 0
    elif parsed.subcommand == "scan":
        stats = run_pipeline(
            input_dir=parsed.path,
            db_path=parsed.db,
            conf_threshold=parsed.conf,
            eps=parsed.threshold,
            clustering_algorithm=parsed.algo,
            export_dir=parsed.export,
            scene_threshold=parsed.scene_thresh,
            min_interval_sec=parsed.min_interval,
            max_interval_sec=parsed.max_interval,
        )
        print_summary(stats)
        return 0
    elif parsed.subcommand == "inspect":
        inspect_cluster_cmd(db_path=parsed.db, cluster_id=parsed.cluster_id)
        return 0
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
