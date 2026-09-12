"""
Command Line Interface for the offline face clustering tool.
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

    if stats["clusters"]:
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


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app",
        description="Offline Photo Face-Clustering Tool POC"
    )

    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommands")

    # Scan subcommand
    scan_parser = subparsers.add_parser("scan", help="Scan a directory, detect faces, and cluster them")
    scan_parser.add_argument("path", type=str, help="Directory path containing photos")
    scan_parser.add_argument("--db", type=str, default="face_clusters.db", help="SQLite database path (default: face_clusters.db)")
    scan_parser.add_argument("--threshold", type=float, default=0.65, help="Cosine distance threshold for clustering (default: 0.65)")
    scan_parser.add_argument("--conf", type=float, default=0.5, help="Face detection confidence threshold (default: 0.5)")
    scan_parser.add_argument("--algo", type=str, default="dbscan", choices=["dbscan", "agglomerative"], help="Clustering algorithm")
    scan_parser.add_argument("--export", type=str, default=None, help="Directory to export cropped face cutouts grouped by person")

    # Inspect subcommand
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a specific person/cluster")
    inspect_parser.add_argument("cluster_id", type=int, help="Cluster ID to inspect")
    inspect_parser.add_argument("--db", type=str, default="face_clusters.db", help="SQLite database path (default: face_clusters.db)")

    # Support default positional path if neither scan nor inspect is explicitly typed
    if args is None:
        args = sys.argv[1:]

    # If first argument is a valid directory or path not matching subcommands, treat as scan command
    if args and args[0] not in ("scan", "inspect", "-h", "--help"):
        # Prepend 'scan'
        args = ["scan"] + args

    parsed = parser.parse_args(args)

    if parsed.subcommand == "scan":
        stats = run_pipeline(
            input_dir=parsed.path,
            db_path=parsed.db,
            conf_threshold=parsed.conf,
            eps=parsed.threshold,
            clustering_algorithm=parsed.algo,
            export_dir=parsed.export,
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
