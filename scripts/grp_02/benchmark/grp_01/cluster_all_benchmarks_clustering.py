"""
Clustering and evaluation functions for cluster_all_benchmarks.py.

Contains similarity matrix computation, optimal cluster finding,
and direction evaluation functions.
"""
import torch
import numpy as np
from typing import List, Dict
from sklearn.cluster import AgglomerativeClustering

def find_optimal_clusters(sim_matrix: np.ndarray, names: List[str], max_clusters: int = 10):
    dist_matrix = 1 - sim_matrix
    best_score, best_n, best_clusters = -1, 2, None
    
    for n_clusters in range(2, min(max_clusters + 1, len(names))):
        try:
            clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
            labels = clustering.fit_predict(dist_matrix)
            
            clusters = {}
            for i, label in enumerate(labels):
                clusters.setdefault(label, []).append(i)
            
            within_sims = []
            for members in clusters.values():
                if len(members) > 1:
                    for i in range(len(members)):
                        for j in range(i+1, len(members)):
                            within_sims.append(sim_matrix[members[i], members[j]])
            
            score = np.mean(within_sims) if within_sims else 0
            if score > best_score:
                best_score, best_n = score, n_clusters
                best_clusters = {k: [names[i] for i in v] for k, v in clusters.items()}
        except:
            pass
    
    return best_n, best_clusters


def evaluate_directions(directions, activations, clusters):
    all_pos = torch.cat([activations[b]['pos'] for b in activations])
    all_neg = torch.cat([activations[b]['neg'] for b in activations])
    global_dir = all_pos.mean(dim=0) - all_neg.mean(dim=0)
    norm = torch.norm(global_dir)
    if norm > 1e-8:
        global_dir = global_dir / norm
    
    cluster_dirs = {}
    bench_to_cluster = {}
    for cid, members in clusters.items():
        valid = [m for m in members if m in activations]
        if valid:
            p = torch.cat([activations[m]['pos'] for m in valid])
            n = torch.cat([activations[m]['neg'] for m in valid])
            d = p.mean(dim=0) - n.mean(dim=0)
            norm = torch.norm(d)
            if norm > 1e-8:
                cluster_dirs[cid] = d / norm
            for m in members:
                bench_to_cluster[m] = cid
    
    global_accs, cluster_accs = [], []
    for bench, acts in activations.items():
        pos, neg = acts['pos'], acts['neg']
        n = min(len(pos), len(neg))
        
        g_correct = sum(1 for i in range(n) if torch.dot(pos[i], global_dir) > torch.dot(neg[i], global_dir))
        global_accs.append(g_correct / n if n > 0 else 0.5)
        
        cid = bench_to_cluster.get(bench)
        if cid in cluster_dirs:
            c_correct = sum(1 for i in range(n) if torch.dot(pos[i], cluster_dirs[cid]) > torch.dot(neg[i], cluster_dirs[cid]))
            cluster_accs.append(c_correct / n if n > 0 else 0.5)
        else:
            cluster_accs.append(global_accs[-1])
    
    return np.mean(global_accs), np.mean(cluster_accs)


def save_and_print_final_results(best_config, all_config_results, MODEL_NAME, model, OUTPUT_DIR, LAYERS, STRATEGIES):
    """Save and print final results."""
    import json
    from dataclasses import asdict
    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"\n{'='*80}")
    logger.info(f"FINAL RESULTS")
    logger.info(f"{'='*80}")

    if best_config:
        logger.info(f"Best: Layer {best_config['layer']}, Strategy {best_config['strategy']}")
        logger.info(f"Global acc: {best_config['global_acc']:.3f}")
        logger.info(f"Cluster acc: {best_config['cluster_acc']:.3f}")

        summary = {
            'model': MODEL_NAME,
            'num_layers': model.config.num_hidden_layers,
            'layers_tested': LAYERS,
            'strategies_tested': STRATEGIES,
            'best_layer': best_config['layer'],
            'best_strategy': best_config['strategy'],
            'n_benchmarks': len(best_config['bench_names']),
            'optimal_clusters': best_config['optimal_n'],
            'global_accuracy': best_config['global_acc'],
            'cluster_accuracy': best_config['cluster_acc'],
            'improvement': best_config['cluster_acc'] - best_config['global_acc'],
            'clusters': best_config['clusters'],
            'combined_geometry': best_config['combined_geometry'],
            'geometry_distribution': best_config['geometry_dist'],
            'all_configs': [asdict(r) for r in all_config_results],
        }

        with open(OUTPUT_DIR / 'cluster_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Save similarity matrix for best config
        with open(OUTPUT_DIR / 'best_similarity_matrix.json', 'w') as f:
            json.dump({
                'bench_names': best_config['bench_names'],
                'sim_matrix': best_config['sim_matrix'],
            }, f, indent=2)

        # Print summary
        print("\n" + "="*80)
        print("BENCHMARK CLUSTERING ANALYSIS - FINAL RESULTS")
        print("="*80)
        print(f"Model: {MODEL_NAME}")
        print(f"Layers tested: {LAYERS}")
        print(f"Strategies tested: {STRATEGIES}")
        print(f"\nBest config: Layer {best_config['layer']}, Strategy {best_config['strategy']}")
        print(f"Benchmarks: {len(best_config['bench_names'])}")
        print(f"Global accuracy: {best_config['global_acc']:.3f}")
        print(f"Cluster accuracy: {best_config['cluster_acc']:.3f}")
        print(f"Improvement: {(best_config['cluster_acc'] - best_config['global_acc'])*100:+.1f}%")
        print(f"\nOptimal clusters: {best_config['optimal_n']}")
        for cid, members in best_config['clusters'].items():
            print(f"  Cluster {cid} ({len(members)} benchmarks): {members[:5]}...")
        print(f"\nCombined geometry: {best_config['combined_geometry']}")
        print(f"Per-benchmark geometry: {best_config['geometry_dist']}")
        print("\nAll configs tested:")
        for r in sorted(all_config_results, key=lambda x: -x.cluster_accuracy):
            print(f"  L{r.layer:2d}_{r.strategy:15s}: global={r.global_accuracy:.3f}, cluster={r.cluster_accuracy:.3f}, geo={r.combined_geometry}")
        print("="*80)
