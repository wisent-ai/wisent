"""K-concept detection algorithm for finding multiple concepts."""

from collections import defaultdict
from typing import Dict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def detect_k_concepts(
    diff_vectors: np.ndarray,
    max_k: int = 6,
    direction_threshold: float = 0.20,  # Clusters with similarity < this are distinct concepts
    seed: int = 42,
) -> Dict:
    """
    Detect the number of distinct concepts in a sample of contrastive pairs.
    
    Method:
    1. For each k from 2 to max_k, cluster into k groups
    2. Compute pairwise direction similarity between all cluster pairs
    3. Count how many cluster pairs have low similarity (< threshold) = distinct concepts
    4. Find k where we have k distinct directions
    
    Args:
        diff_vectors: [N, hidden_dim] difference vectors
        max_k: Maximum number of clusters to try
        direction_threshold: Below this similarity, clusters are considered distinct concepts
        seed: Random seed
        
    Returns:
        Dictionary with detected number of concepts and analysis details
    """
    n = len(diff_vectors)
    max_k = min(max_k, n // 5)  # Need at least 5 samples per cluster
    
    results_by_k = {}
    
    for k in range(2, max_k + 1):
        # Cluster
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(diff_vectors)
        
        # Compute direction for each cluster
        directions = []
        cluster_sizes = []
        for cluster_id in range(k):
            mask = labels == cluster_id
            cluster_size = mask.sum()
            cluster_sizes.append(cluster_size)
            
            if cluster_size >= 3:
                direction = diff_vectors[mask].mean(axis=0)
                direction = direction / (np.linalg.norm(direction) + 1e-10)
                directions.append(direction)
            else:
                directions.append(None)
        
        # Compute pairwise similarities
        pairwise_sims = []
        distinct_pairs = []
        
        for i in range(k):
            for j in range(i + 1, k):
                if directions[i] is not None and directions[j] is not None:
                    sim = np.abs(np.dot(directions[i], directions[j]))
                    pairwise_sims.append({
                        'clusters': (i, j),
                        'similarity': sim,
                        'is_distinct': sim < direction_threshold
                    })
                    if sim < direction_threshold:
                        distinct_pairs.append((i, j))
        
        # Compute silhouette
        if k < n:
            sil = silhouette_score(diff_vectors, labels)
        else:
            sil = 0
        
        # Count distinct concepts using graph connectivity
        # If clusters A-B are distinct and B-C are distinct, we have 3 concepts
        # Build adjacency: clusters are "same concept" if similarity > threshold
        from collections import defaultdict
        
        same_concept_graph = defaultdict(set)
        for i in range(k):
            same_concept_graph[i].add(i)  # Self
            
        for pair_info in pairwise_sims:
            i, j = pair_info['clusters']
            if not pair_info['is_distinct']:  # High similarity = same concept
                same_concept_graph[i].add(j)
                same_concept_graph[j].add(i)
        
        # Find connected components (each component = one concept)
        visited = set()
        num_distinct_concepts = 0
        concept_groups = []
        
        for start in range(k):
            if start not in visited:
                # BFS to find component
                component = set()
                queue = [start]
                while queue:
                    node = queue.pop(0)
                    if node not in visited:
                        visited.add(node)
                        component.add(node)
                        for neighbor in same_concept_graph[node]:
                            if neighbor not in visited:
                                queue.append(neighbor)
                concept_groups.append(component)
                num_distinct_concepts += 1
        
        # Compute average within-concept similarity and between-concept similarity
        within_sims = []
        between_sims = []
        for pair_info in pairwise_sims:
            i, j = pair_info['clusters']
            # Check if i and j are in the same concept group
            same_group = any(i in group and j in group for group in concept_groups)
            if same_group:
                within_sims.append(pair_info['similarity'])
            else:
                between_sims.append(pair_info['similarity'])
        
        results_by_k[k] = {
            'num_clusters': k,
            'num_distinct_concepts': num_distinct_concepts,
            'concept_groups': [list(g) for g in concept_groups],
            'cluster_sizes': cluster_sizes,
            'silhouette': sil,
            'pairwise_similarities': pairwise_sims,
            'num_distinct_pairs': len(distinct_pairs),
            'avg_within_concept_sim': np.mean(within_sims) if within_sims else 1.0,
            'avg_between_concept_sim': np.mean(between_sims) if between_sims else 0.0,
            'min_pairwise_sim': min(p['similarity'] for p in pairwise_sims) if pairwise_sims else 1.0,
        }
    
    # Determine optimal k
    # Look for the k where num_distinct_concepts stabilizes
    # and silhouette is reasonable
    
    concept_counts = [(k, results_by_k[k]['num_distinct_concepts']) for k in range(2, max_k + 1)]
    
    # Find the smallest k where we capture all distinct concepts
    # (where increasing k doesn't increase num_distinct_concepts)
    optimal_k = 2
    optimal_concepts = results_by_k[2]['num_distinct_concepts']
    
    for k in range(2, max_k + 1):
        current_concepts = results_by_k[k]['num_distinct_concepts']
        current_sil = results_by_k[k]['silhouette']
        
        # Accept this k if it finds more concepts and has decent silhouette
        if current_concepts > optimal_concepts and current_sil > 0.05:
            optimal_k = k
            optimal_concepts = current_concepts
    
    return {
        'detected_concepts': optimal_concepts,
        'optimal_k': optimal_k,
        'results_by_k': results_by_k,
        'recommendation': f"Detected {optimal_concepts} distinct concept(s) using k={optimal_k} clusters",
    }

