import numpy as np
import cupy as cp
import random
import tiktoken
from cuml.manifold import UMAP  # GPU-based UMAP
from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed
from typing import Dict, List, Optional

from llama_index.core.schema import BaseNode

# Set a random seed for reproducibility
RANDOM_SEED = 224
random.seed(RANDOM_SEED)


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    print("-----------------Optimized GPU clustering-----------------")
    # Move embeddings to GPU once
    embeddings_gpu = cp.asarray(embeddings)
    if n_neighbors is None:
        n_neighbors = min(15, embeddings.shape[0] - 1)
    umap_model = UMAP(
        n_neighbors=n_neighbors, 
        n_components=dim, 
        metric=metric,
        method="graph",  # Enables faster neighbor search
        verbose=True
    )
    # Result remains a CuPy array; convert to NumPy if needed downstream
    return cp.asnumpy(umap_model.fit_transform(embeddings_gpu))


def local_cluster_embeddings(
    embeddings: np.ndarray, 
    dim: int, 
    num_neighbors: int = 10, 
    metric: str = "cosine"
) -> np.ndarray:
    embeddings_gpu = cp.asarray(embeddings)
    umap_model = UMAP(
        n_neighbors=num_neighbors, 
        n_components=dim, 
        metric=metric,
        method="graph", 
        verbose=True
    )
    return cp.asnumpy(umap_model.fit_transform(embeddings_gpu))


def bic_for_n(n: int, embeddings_np: np.ndarray, random_state: int) -> float:
    gm = GaussianMixture(n_components=n, random_state=random_state)
    gm.fit(embeddings_np)
    return gm.bic(embeddings_np)


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    # Limit max clusters to the number of embeddings available
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters_candidates = np.arange(1, max_clusters)
    # Ensure embeddings are in NumPy format
    embeddings_np = embeddings.get() if isinstance(embeddings, cp.ndarray) else embeddings
    # Parallelize the BIC computations over candidate cluster counts
    bics = Parallel(n_jobs=-1)(
        delayed(bic_for_n)(n, embeddings_np, random_state) for n in n_clusters_candidates
    )
    optimal_n = n_clusters_candidates[np.argmin(bics)]
    return optimal_n


def GMM_cluster(
    embeddings: np.ndarray, threshold: float, random_state: int = 0
):
    # Ensure embeddings are on CPU for GaussianMixture
    embeddings_np = embeddings.get() if isinstance(embeddings, cp.ndarray) else embeddings
    n_clusters = get_optimal_clusters(embeddings_np, random_state=random_state)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings_np)
    probs = gm.predict_proba(embeddings_np)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray,
    dim: int,
    threshold: float,
) -> List[np.ndarray]:
    # If too few embeddings, assign all to the same cluster
    if len(embeddings) <= dim + 1:
        return [np.array([0]) for _ in range(len(embeddings))]
    
    # Global dimensionality reduction using GPU UMAP
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    for i in range(n_global_clusters):
        # Select embeddings in global cluster i
        global_cluster_mask = np.array([i in gc for gc in global_clusters])
        global_cluster_embeddings_ = embeddings[global_cluster_mask]

        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in range(len(global_cluster_embeddings_))]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        for j in range(n_local_clusters):
            # Identify indices for local cluster j within the global cluster
            local_mask = np.array([j in lc for lc in local_clusters])
            local_cluster_embeddings_ = global_cluster_embeddings_[local_mask]
            # Map these local embeddings back to indices in the original embeddings array.
            # This uses a vectorized comparison; adjust tolerance as needed.
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )
        total_clusters += n_local_clusters

    return all_local_clusters


def get_clusters(
    nodes: List[BaseNode],
    embedding_map: Dict[str, List[List[float]]],
    max_length_in_cluster: int = 10000,  # 10k tokens max per cluster
    tokenizer: tiktoken.Encoding = tiktoken.get_encoding("cl100k_base"),
    reduction_dimension: int = 10,
    threshold: float = 0.1,
    prev_total_length=None,  # Track previous total length to prevent infinite recursion
) -> List[List[BaseNode]]:
    # Assemble embeddings as a NumPy array
    embeddings = np.array([np.array(embedding_map[node.id_]) for node in nodes])

    # Perform the clustering
    clusters = perform_clustering(
        embeddings, dim=reduction_dimension, threshold=threshold
    )

    node_clusters = []
    # Process each unique label to form clusters of nodes
    for label in np.unique(np.concatenate(clusters)):
        indices = [i for i, cluster in enumerate(clusters) if label in cluster]
        cluster_nodes = [nodes[i] for i in indices]

        if len(cluster_nodes) == 1:
            node_clusters.append(cluster_nodes)
            continue

        total_length = sum([len(tokenizer.encode(node.text)) for node in cluster_nodes])
        if total_length > max_length_in_cluster and (
            prev_total_length is None or total_length < prev_total_length
        ):
            node_clusters.extend(
                get_clusters(
                    cluster_nodes,
                    embedding_map,
                    max_length_in_cluster=max_length_in_cluster,
                    tokenizer=tokenizer,
                    reduction_dimension=reduction_dimension,
                    threshold=threshold,
                    prev_total_length=total_length,
                )
            )
        else:
            node_clusters.append(cluster_nodes)

    return node_clusters