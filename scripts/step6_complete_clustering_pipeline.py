from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import KNeighborsClassifier


K_NEIGHBORS = 40
OUTPUT_DIR_NAME = "knn_sparsification_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complete clustering pipeline with sparse KNN graph.")
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Path to experiment directory (e.g., GeoRFExperiment)",
    )
    parser.add_argument("--n-clusters", type=int, default=None, help="Number of clusters")
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to knn_analysis_report_k40.json for auto cluster count",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="general",
        help="Suffix for output naming (general, m02, m06, m10)",
    )
    parser.add_argument(
        "--similarity-dir",
        type=str,
        default="similarity_matrices",
        help="Directory containing similarity_matrices*.npz",
    )
    return parser.parse_args()


def resolve_similarity_npz(base_dir: Path, similarity_dir_arg: str) -> Path:
    similarity_path = Path(similarity_dir_arg)
    if not similarity_path.is_absolute():
        similarity_path = base_dir / similarity_path

    if similarity_path.is_file() and similarity_path.suffix == ".npz":
        return similarity_path

    if not similarity_path.exists() or not similarity_path.is_dir():
        raise FileNotFoundError(f"Similarity dir not found: {similarity_path}")

    files = sorted(similarity_path.glob("similarity_matrices*.npz"))
    if not files:
        raise FileNotFoundError(f"No similarity matrix npz found in: {similarity_path}")
    return files[0]


def resolve_report_path(base_dir: Path, report_arg: Optional[str], suffix: str = "general") -> Path:
    if report_arg is None:
        return base_dir / OUTPUT_DIR_NAME / f"knn_analysis_report_k{K_NEIGHBORS}_{suffix}.json"
    report_path = Path(report_arg)
    if not report_path.is_absolute():
        report_path = base_dir / report_path
    return report_path


def determine_n_clusters(n_clusters_arg: Optional[int], report_path: Path) -> int:
    if n_clusters_arg is not None:
        if n_clusters_arg <= 0:
            raise ValueError("--n-clusters must be positive")
        return int(n_clusters_arg)

    if not report_path.exists():
        raise FileNotFoundError(
            f"No --n-clusters provided and report not found: {report_path}"
        )

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    if "recommended_clusters" not in report:
        raise KeyError(
            f"Report missing 'recommended_clusters': {report_path}"
        )

    recommended = int(report["recommended_clusters"])
    if recommended <= 0:
        raise ValueError(f"Invalid recommended_clusters in report: {recommended}")
    return recommended


_SUFFIX_TO_MANIFEST_KEY: dict[str, str] = {
    "general": "general",
    "m2": "m02",
    "m6": "m06",
    "m10": "m10",
}


def update_manifest(base_dir: Path, suffix: str, mapping_path: Path, n_clusters: int) -> None:
    manifest_path = base_dir / OUTPUT_DIR_NAME / "cluster_mapping_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {}

    for key in ["general", "m02", "m06", "m10"]:
        manifest.setdefault(key, None)

    entry = {
        "path": str(mapping_path),
        "n_clusters": int(n_clusters),
    }
    # Normalize suffix to padded manifest key (m2 -> m02, m6 -> m06)
    manifest_key = _SUFFIX_TO_MANIFEST_KEY.get(suffix, suffix)
    manifest[manifest_key] = entry

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Updated manifest: {manifest_path} (key={manifest_key})")


def main() -> int:
    args = parse_args()

    try:
        base_dir = Path(args.experiment_dir).expanduser().resolve()
        if not base_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {base_dir}")

        output_dir = base_dir / OUTPUT_DIR_NAME
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = resolve_report_path(base_dir, args.report, args.suffix)
        n_clusters = determine_n_clusters(args.n_clusters, report_path)
        similarity_npz = resolve_similarity_npz(base_dir, args.similarity_dir)

        print(f"Using n_clusters={n_clusters}")
        print(f"Using similarity file: {similarity_npz}")

        sparse_affinity = load_npz(output_dir / f"knn_graph_k{K_NEIGHBORS}_{args.suffix}.npz")
        comp_data = np.load(output_dir / f"connected_components_k{K_NEIGHBORS}_{args.suffix}.npz")
        component_labels = comp_data["labels"]
        n = sparse_affinity.shape[0]

        matrix_data = np.load(similarity_npz)
        if "lat_lon_data" not in matrix_data:
            raise KeyError(f"lat_lon_data missing from {similarity_npz}")
        lat_lon_data = matrix_data["lat_lon_data"]

        counts = np.bincount(component_labels)
        main_component_id = int(np.argmax(counts))
        main_indices = np.where(component_labels == main_component_id)[0]
        outlier_indices = np.where(component_labels != main_component_id)[0]

        print(f"Main component: {len(main_indices)} nodes")
        print(f"Outliers: {len(outlier_indices)} nodes")

        sub_affinity = sparse_affinity[main_indices][:, main_indices]

        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=42,
            n_jobs=-1,
        )
        main_labels = sc.fit_predict(sub_affinity)

        if len(outlier_indices) > 0:
            knn_classifier = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
            knn_classifier.fit(lat_lon_data[main_indices], main_labels)
            outlier_labels = knn_classifier.predict(lat_lon_data[outlier_indices])
        else:
            outlier_labels = None

        final_labels = np.zeros(n, dtype=int)
        final_labels[main_indices] = main_labels
        if outlier_labels is not None:
            final_labels[outlier_indices] = outlier_labels

        labels_npz = output_dir / f"final_cluster_labels_k{K_NEIGHBORS}_nc{n_clusters}_{args.suffix}.npz"
        np.savez_compressed(
            labels_npz,
            cluster_labels=final_labels,
            admin_codes=np.arange(n),
            n_clusters=np.int32(n_clusters),
            k_neighbors=np.int32(K_NEIGHBORS),
            main_indices=main_indices,
            outlier_indices=outlier_indices,
        )
        print(f"Saved labels: {labels_npz}")

        mapping_file = output_dir / f"cluster_mapping_k{K_NEIGHBORS}_nc{n_clusters}_{args.suffix}.csv"
        cluster_mapping = pd.DataFrame(
            {
                "FEWSNET_admin_code": np.arange(n),
                "cluster_id": final_labels,
                "latitude": lat_lon_data[:, 0],
                "longitude": lat_lon_data[:, 1],
                "is_outlier": np.isin(np.arange(n), outlier_indices),
            }
        )
        cluster_mapping.to_csv(mapping_file, index=False)
        print(f"Saved mapping: {mapping_file}")

        report_payload = {}
        if report_path.exists():
            with report_path.open("r", encoding="utf-8") as f:
                report_payload = json.load(f)

        unique_final, counts_final = np.unique(final_labels, return_counts=True)
        report_payload["spectral_clustering"] = {
            "n_clusters": int(n_clusters),
            "suffix": args.suffix,
            "main_component_size": int(len(main_indices)),
            "n_outliers": int(len(outlier_indices)),
            "cluster_sizes": {int(k): int(v) for k, v in zip(unique_final, counts_final)},
            "method": "spectral_clustering_with_outlier_knn",
        }

        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2)
        print(f"Updated report: {report_path}")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        scatter = axes[0].scatter(
            lat_lon_data[:, 1],
            lat_lon_data[:, 0],
            c=final_labels,
            cmap="tab10",
            s=5,
            alpha=0.6,
        )
        axes[0].set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")
        axes[0].set_title(f"Geographic Clusters ({args.suffix})")
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0], label="Cluster ID")

        axes[1].bar(unique_final, counts_final, color="steelblue", edgecolor="black", alpha=0.7)
        axes[1].set_xlabel("Cluster ID")
        axes[1].set_ylabel("Number of Admin Codes")
        axes[1].set_title("Cluster Sizes")
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plot_file = output_dir / f"final_clustering_results_{args.suffix}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {plot_file}")

        update_manifest(base_dir, args.suffix, mapping_file, n_clusters)
        print("Step 6 complete clustering completed")
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
