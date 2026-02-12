from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, diags, save_npz
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh


K_NEIGHBORS = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparsify similarity matrix with KNN graph.")
    parser.add_argument("--experiment-dir", required=True, help="Experiment directory path")
    parser.add_argument(
        "--similarity-dir",
        default="similarity_matrices",
        help="Directory containing similarity_matrices*.npz",
    )
    parser.add_argument(
        "--suffix",
        default="general",
        help="Run suffix for output naming (general, m2, m6, m10)",
    )
    return parser.parse_args()


def resolve_npz_path(experiment_dir: Path, similarity_dir: str) -> Path:
    target = Path(similarity_dir)
    if not target.is_absolute():
        target = experiment_dir / target

    if target.is_file() and target.suffix == ".npz":
        return target

    if not target.exists() or not target.is_dir():
        raise FileNotFoundError(f"Similarity directory not found: {target}")

    candidates = sorted(target.glob("similarity_matrices*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No similarity matrix npz found in: {target}")
    selected = candidates[0]
    return selected


def build_knn_similarity_graph(matrix: NDArray[np.float32], k: int) -> csr_matrix:
    n = matrix.shape[0]
    data = []
    rows = []
    cols = []

    for i in range(n):
        row_values = matrix[i]
        top_indices = np.argpartition(row_values, -k)[-k:] if n > k else np.arange(n)
        rows.extend([i] * len(top_indices))
        cols.extend(top_indices.tolist())
        data.extend(row_values[top_indices].tolist())
        if (i + 1) % 1000 == 0 or (i + 1) == n:
            print(f"  KNN processed rows: {i + 1}/{n}")

    knn_mat = csr_matrix((data, (rows, cols)), shape=(n, n))
    return knn_mat.maximum(knn_mat.transpose())


def compute_eigengap(
    affinity_mat: csr_matrix, n_eigenvalues: int = 20
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    n, _ = affinity_mat.get_shape()
    n = int(n)
    k = max(2, min(n_eigenvalues, n - 2))

    degrees = np.asarray(affinity_mat.sum(axis=1)).reshape(-1).astype(np.float64, copy=False)
    degrees = np.where(degrees == 0, 1e-10, degrees)
    d_inv_sqrt = diags(1.0 / np.sqrt(degrees))
    identity = diags(np.ones(n, dtype=np.float64))
    laplacian = identity - d_inv_sqrt @ affinity_mat @ d_inv_sqrt

    eigenvalues, _ = eigsh(laplacian, k=k, which="SM", maxiter=5000)
    eigenvalues = np.sort(eigenvalues)
    eigengaps = np.diff(eigenvalues)
    if eigengaps.size == 0:
        raise RuntimeError("Not enough eigenvalues for eigengap analysis")
    max_gap_idx = int(np.argmax(eigengaps))
    recommended_clusters = max_gap_idx + 1
    return eigenvalues, eigengaps, recommended_clusters


def save_knn_plots(
    output_dir: Path,
    knn_graph: csr_matrix,
    n_components: int,
    component_labels: NDArray[np.int32],
    eigenvalues: NDArray[np.float64],
    eigengaps: NDArray[np.float64],
    suffix: str = "general",
) -> None:
    degrees = np.asarray(knn_graph.sum(axis=1)).ravel()
    edge_weights = knn_graph.data

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(degrees, bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_title(f"Degree Distribution (k={K_NEIGHBORS})")
    axes[0].set_xlabel("Weighted degree")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(edge_weights, bins=50, edgecolor="black", alpha=0.7)
    axes[1].set_yscale("log")
    axes[1].set_title("Edge Weight Distribution")
    axes[1].set_xlabel("Weight")
    axes[1].set_ylabel("Count (log)")
    axes[1].grid(True, alpha=0.3)

    if n_components > 1:
        _, counts = np.unique(component_labels, return_counts=True)
        axes[2].hist(counts, bins=50, edgecolor="black", alpha=0.7)
        axes[2].set_yscale("log")
        axes[2].set_title(f"Component Sizes ({n_components} components)")
        axes[2].set_xlabel("Component size")
        axes[2].set_ylabel("Count (log)")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "Graph is fully connected", ha="center", va="center")
        axes[2].axis("off")
        axes[2].set_title("Connectivity")

    fig.tight_layout()
    fig.savefig(output_dir / f"knn_graph_analysis_k{K_NEIGHBORS}_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    x_vals = np.arange(1, len(eigenvalues) + 1)
    axes2[0].plot(x_vals, eigenvalues, "o-", linewidth=2)
    axes2[0].set_title("Normalized Laplacian Eigenvalues")
    axes2[0].set_xlabel("Index")
    axes2[0].set_ylabel("Eigenvalue")
    axes2[0].grid(True, alpha=0.3)

    gap_x = np.arange(1, len(eigengaps) + 1)
    max_gap_idx = int(np.argmax(eigengaps))
    axes2[1].bar(gap_x, eigengaps, edgecolor="black", alpha=0.7)
    axes2[1].bar(max_gap_idx + 1, eigengaps[max_gap_idx], color="red", edgecolor="black")
    axes2[1].set_title("Eigengap Analysis")
    axes2[1].set_xlabel("Gap index")
    axes2[1].set_ylabel("Gap size")
    axes2[1].grid(True, alpha=0.3, axis="y")

    fig2.tight_layout()
    fig2.savefig(output_dir / f"eigengap_analysis_k{K_NEIGHBORS}_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)


def main() -> int:
    args = parse_args()
    try:
        experiment_dir = Path(args.experiment_dir).expanduser().resolve()
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

        matrix_file = resolve_npz_path(experiment_dir, args.similarity_dir)
        print(f"Loading similarity matrices from: {matrix_file}")
        data = np.load(matrix_file)
        if "final_matrix" not in data:
            raise ValueError(f"NPZ file missing key 'final_matrix': {matrix_file}")

        final_matrix = data["final_matrix"].astype(np.float32)
        n = final_matrix.shape[0]
        print(f"Final matrix shape: {final_matrix.shape}")

        print(f"Building KNN graph with k={K_NEIGHBORS}")
        sparse_affinity = build_knn_similarity_graph(final_matrix, K_NEIGHBORS)

        n_components, labels = connected_components(
            csgraph=sparse_affinity, directed=False, return_labels=True
        )
        print(f"Connected components: {n_components}")

        eigenvalues, eigengaps, recommended_clusters = compute_eigengap(sparse_affinity, 20)
        print(f"Recommended clusters from eigengap: {recommended_clusters}")

        component_ids, component_sizes = np.unique(labels, return_counts=True)
        largest_component_size = int(component_sizes.max()) if component_sizes.size else 0
        n_outliers = int(n - largest_component_size)

        output_dir = experiment_dir / "knn_sparsification_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        suffix = args.suffix
        save_npz(output_dir / f"knn_graph_k{K_NEIGHBORS}_{suffix}.npz", sparse_affinity)
        np.savez_compressed(
            output_dir / f"connected_components_k{K_NEIGHBORS}_{suffix}.npz",
            labels=labels,
            n_components=np.int32(n_components),
            component_ids=component_ids,
            component_sizes=component_sizes,
        )

        report = {
            "recommended_clusters": int(recommended_clusters),
            "knn_parameters": {
                "k_neighbors": int(K_NEIGHBORS),
                "graph_size": int(n),
                "n_edges": int(sparse_affinity.nnz),
                "density": float(sparse_affinity.nnz / (n * max(1, n - 1))),
                "sparsity": float(1.0 - sparse_affinity.nnz / (n * n)),
            },
            "connectivity": {
                "n_components": int(n_components),
                "largest_component_size": int(largest_component_size),
                "n_outliers": int(n_outliers),
                "largest_component_pct": float(100.0 * largest_component_size / n),
            },
            "eigengap_analysis": {
                "n_eigenvalues_computed": int(len(eigenvalues)),
                "eigenvalues": eigenvalues.tolist(),
                "eigengaps": eigengaps.tolist(),
                "suggested_n_clusters": int(recommended_clusters),
            },
            "source_similarity_npz": str(matrix_file),
        }
        report_path = output_dir / f"knn_analysis_report_k{K_NEIGHBORS}_{suffix}.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        save_knn_plots(output_dir, sparse_affinity, n_components, labels, eigenvalues, eigengaps, suffix)

        print(f"Saved: {output_dir / f'knn_graph_k{K_NEIGHBORS}_{suffix}.npz'}")
        print(f"Saved: {output_dir / f'connected_components_k{K_NEIGHBORS}_{suffix}.npz'}")
        print(f"Saved: {report_path}")
        print(f"Saved plots: knn_graph_analysis_k{K_NEIGHBORS}_{suffix}.png, eigengap_analysis_k{K_NEIGHBORS}_{suffix}.png")
        print(f"Step 5 sparsification completed (suffix={suffix})")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
