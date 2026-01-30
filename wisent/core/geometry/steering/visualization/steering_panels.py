"""Individual panel plotting functions for steering visualization."""

import numpy as np
import warnings


def get_eval_colors(evaluations, n_points):
    """Get edge colors based on evaluations."""
    if evaluations is None:
        return ['black'] * n_points
    colors = []
    for i in range(n_points):
        if i < len(evaluations):
            colors.append('green' if evaluations[i] == "TRUTHFUL" else 'red')
        else:
            colors.append('black')
    return colors


def plot_projection_panel(ax, pos_2d, neg_2d, base_2d, steered_2d,
                          base_evals, steered_evals, title):
    """Common projection plotting logic."""
    ax.scatter(pos_2d[:, 0], pos_2d[:, 1], c='blue', alpha=0.3, s=30, label='Positive')
    ax.scatter(neg_2d[:, 0], neg_2d[:, 1], c='red', alpha=0.3, s=30, label='Negative')

    pos_centroid = pos_2d.mean(axis=0)
    neg_centroid = neg_2d.mean(axis=0)
    ax.scatter([pos_centroid[0]], [pos_centroid[1]], c='blue', s=150, marker='*',
               edgecolors='black', linewidths=1, zorder=5)
    ax.scatter([neg_centroid[0]], [neg_centroid[1]], c='red', s=150, marker='*',
               edgecolors='black', linewidths=1, zorder=5)

    base_colors = get_eval_colors(base_evals, len(base_2d))
    for i, (x, y) in enumerate(base_2d):
        ax.scatter([x], [y], c='gray', s=60, marker='o',
                   edgecolors=base_colors[i], linewidths=2, zorder=4)

    steered_colors = get_eval_colors(steered_evals, len(steered_2d))
    for i, (x, y) in enumerate(steered_2d):
        ax.scatter([x], [y], c='lime', s=60, marker='s',
                   edgecolors=steered_colors[i], linewidths=2, zorder=4)

    for i in range(min(len(base_2d), len(steered_2d))):
        ax.annotate('', xy=(steered_2d[i, 0], steered_2d[i, 1]),
                    xytext=(base_2d[i, 0], base_2d[i, 1]),
                    arrowprops=dict(arrowstyle='->', color='green', alpha=0.5, lw=1))

    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_pca_panel(ax, pos, neg, base, steered, base_evals, steered_evals):
    """PCA projection panel."""
    from sklearn.decomposition import PCA
    reference = np.vstack([pos, neg])
    pca = PCA(n_components=2, random_state=42)
    pca.fit(reference)
    var_explained = sum(pca.explained_variance_ratio_) * 100
    plot_projection_panel(ax, pca.transform(pos), pca.transform(neg),
                          pca.transform(base), pca.transform(steered),
                          base_evals, steered_evals, f"PCA ({var_explained:.1f}% var)")


def plot_lda_panel(ax, pos, neg, base, steered, base_evals, steered_evals):
    """LDA projection panel."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.decomposition import PCA
    reference = np.vstack([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    try:
        lda = LinearDiscriminantAnalysis(n_components=1)
        lda.fit(reference, labels)
        lda_ref = lda.transform(reference)
        pca = PCA(n_components=1, random_state=42)
        residual = reference - (lda_ref @ lda.scalings_.T + lda.xbar_)
        pca.fit(residual)
        ref_2d = np.hstack([lda_ref, pca.transform(residual)])

        base_lda = lda.transform(base)
        base_res = base - (base_lda @ lda.scalings_.T + lda.xbar_)
        base_2d = np.hstack([base_lda, pca.transform(base_res)])

        steered_lda = lda.transform(steered)
        steered_res = steered - (steered_lda @ lda.scalings_.T + lda.xbar_)
        steered_2d = np.hstack([steered_lda, pca.transform(steered_res)])

        plot_projection_panel(ax, ref_2d[:len(pos)], ref_2d[len(pos):],
                              base_2d, steered_2d, base_evals, steered_evals, "LDA + PCA")
    except Exception:
        ax.text(0.5, 0.5, "LDA error", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("LDA (failed)")


def plot_tsne_panel(ax, pos, neg, base, steered, base_evals, steered_evals):
    """t-SNE projection panel."""
    from sklearn.manifold import TSNE
    all_data = np.vstack([pos, neg, base, steered])
    n = len(all_data)
    if n < 5:
        ax.text(0.5, 0.5, "Not enough samples", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("t-SNE (failed)")
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tsne = TSNE(n_components=2, perplexity=min(30, n-1), random_state=42)
        all_2d = tsne.fit_transform(all_data)
    n_pos, n_neg = len(pos), len(neg)
    plot_projection_panel(ax, all_2d[:n_pos], all_2d[n_pos:n_pos+n_neg],
                          all_2d[n_pos+n_neg:n_pos+n_neg+len(base)],
                          all_2d[n_pos+n_neg+len(base):], base_evals, steered_evals, "t-SNE")


def plot_umap_panel(ax, pos, neg, base, steered, base_evals, steered_evals):
    """UMAP projection panel."""
    try:
        import umap
    except ImportError:
        ax.text(0.5, 0.5, "UMAP not installed", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("UMAP (unavailable)")
        return
    all_data = np.vstack([pos, neg, base, steered])
    n = len(all_data)
    if n < 5:
        ax.text(0.5, 0.5, "Not enough samples", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("UMAP (failed)")
        return
    reducer = umap.UMAP(n_components=2, n_neighbors=min(15, n-1), min_dist=0.1, random_state=42)
    all_2d = reducer.fit_transform(all_data)
    n_pos, n_neg = len(pos), len(neg)
    plot_projection_panel(ax, all_2d[:n_pos], all_2d[n_pos:n_pos+n_neg],
                          all_2d[n_pos+n_neg:n_pos+n_neg+len(base)],
                          all_2d[n_pos+n_neg+len(base):], base_evals, steered_evals, "UMAP")


def plot_pacmap_panel(ax, pos, neg, base, steered, base_evals, steered_evals):
    """PaCMAP projection panel using pacmap_alt."""
    try:
        from .pacmap_alt import pacmap_embedding
        all_data = np.vstack([pos, neg, base, steered])
        n = len(all_data)
        if n < 5:
            raise ValueError("Not enough samples")
        all_2d = pacmap_embedding(all_data, n_components=2, n_neighbors=min(10, n//4), num_iters=50)
        n_pos, n_neg = len(pos), len(neg)
        plot_projection_panel(ax, all_2d[:n_pos], all_2d[n_pos:n_pos+n_neg],
                              all_2d[n_pos+n_neg:n_pos+n_neg+len(base)],
                              all_2d[n_pos+n_neg+len(base):], base_evals, steered_evals, "PaCMAP")
    except Exception as e:
        ax.text(0.5, 0.5, f"PaCMAP: {str(e)[:25]}", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("PaCMAP (failed)")


def plot_pca_with_boundary(ax, pos, neg, base, steered, base_evals, steered_evals):
    """PCA with decision boundary visualization."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    reference = np.vstack([pos, neg])
    pca = PCA(n_components=2, random_state=42)
    pca.fit(reference)
    pos_2d, neg_2d = pca.transform(pos), pca.transform(neg)
    base_2d, steered_2d = pca.transform(base), pca.transform(steered)

    X_2d = np.vstack([pos_2d, neg_2d])
    y_2d = np.concatenate([np.ones(len(pos_2d)), np.zeros(len(neg_2d))])
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_2d, y_2d)

    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['#FFCCCC', '#CCCCFF'], alpha=0.4)
    ax.contour(xx, yy, Z, levels=[0.5], colors=['black'], linewidths=2, linestyles=['--'])
    plot_projection_panel(ax, pos_2d, neg_2d, base_2d, steered_2d,
                          base_evals, steered_evals, "PCA + Decision Boundary")


def plot_movement_vectors(ax, pos, neg, base, steered):
    """Movement vector analysis panel."""
    from sklearn.decomposition import PCA
    n = min(len(base), len(steered))
    movements = steered[:n] - base[:n]
    mean_steering = pos.mean(axis=0) - neg.mean(axis=0)
    mean_steering_norm = mean_steering / (np.linalg.norm(mean_steering) + 1e-8)

    if len(movements) > 2:
        pca = PCA(n_components=2, random_state=42)
        movements_2d = pca.fit_transform(movements)
        mean_steering_2d = pca.transform(mean_steering.reshape(1, -1))[0]

        ax.scatter(movements_2d[:, 0], movements_2d[:, 1], c='green', alpha=0.6, s=40)
        ax.scatter([0], [0], c='black', s=100, marker='x', zorder=5)

        scale = np.max(np.abs(movements_2d)) * 0.8
        norm_ms = np.linalg.norm(mean_steering_2d) + 1e-8
        ax.arrow(0, 0, mean_steering_2d[0]*scale/norm_ms, mean_steering_2d[1]*scale/norm_ms,
                 head_width=scale*0.1, head_length=scale*0.05, fc='red', ec='red')

        movement_norms = np.linalg.norm(movements, axis=1)
        valid = movement_norms > 1e-8
        alignments = np.zeros(n)
        alignments[valid] = (movements[valid] / movement_norms[valid, np.newaxis]) @ mean_steering_norm
        mean_align = alignments[valid].mean() if valid.any() else 0
        ax.set_title(f"Movement Vectors (align={mean_align:.2f})")
    else:
        ax.text(0.5, 0.5, "Not enough samples", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Movement Vectors")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")


def plot_norm_distribution(ax, pos, neg, base, steered):
    """Norm distribution panel."""
    pos_norms = np.linalg.norm(pos, axis=1)
    neg_norms = np.linalg.norm(neg, axis=1)
    base_norms = np.linalg.norm(base, axis=1)
    steered_norms = np.linalg.norm(steered, axis=1)

    bins = np.linspace(
        min(pos_norms.min(), neg_norms.min(), base_norms.min(), steered_norms.min()),
        max(pos_norms.max(), neg_norms.max(), base_norms.max(), steered_norms.max()), 30)

    ax.hist(pos_norms, bins=bins, alpha=0.4, label='Positive', color='blue')
    ax.hist(neg_norms, bins=bins, alpha=0.4, label='Negative', color='red')
    ax.hist(base_norms, bins=bins, alpha=0.6, label='Base', color='gray', histtype='step', linewidth=2)
    ax.hist(steered_norms, bins=bins, alpha=0.6, label='Steered', color='green', histtype='step', linewidth=2)
    ax.set_title("Norm Distribution")
    ax.set_xlabel("L2 Norm")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_alignment_histogram(ax, pos, neg, base, steered):
    """Alignment with steering direction histogram."""
    mean_steering = pos.mean(axis=0) - neg.mean(axis=0)
    mean_steering_norm = mean_steering / (np.linalg.norm(mean_steering) + 1e-8)

    def compute_alignments(data):
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        valid = norms.squeeze() > 1e-8
        normalized = np.zeros_like(data)
        normalized[valid] = data[valid] / norms[valid]
        return normalized @ mean_steering_norm

    base_align = compute_alignments(base)
    steered_align = compute_alignments(steered)
    bins = np.linspace(-1, 1, 30)
    ax.hist(base_align, bins=bins, alpha=0.6, label='Base', color='gray')
    ax.hist(steered_align, bins=bins, alpha=0.6, label='Steered', color='green')
    ax.axvline(base_align.mean(), color='gray', linestyle='--', linewidth=2)
    ax.axvline(steered_align.mean(), color='green', linestyle='--', linewidth=2)
    ax.set_title("Alignment with Steering Direction")
    ax.set_xlabel("Cosine Alignment")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_centroid_distances(ax, pos, neg, base, steered):
    """Distance to centroids panel."""
    pos_centroid = pos.mean(axis=0)
    neg_centroid = neg.mean(axis=0)
    base_to_pos = np.linalg.norm(base - pos_centroid, axis=1)
    base_to_neg = np.linalg.norm(base - neg_centroid, axis=1)
    steered_to_pos = np.linalg.norm(steered - pos_centroid, axis=1)
    steered_to_neg = np.linalg.norm(steered - neg_centroid, axis=1)

    x = np.arange(len(base))
    width = 0.35
    ax.bar(x - width/2, base_to_pos - base_to_neg, width, label='Base', color='gray', alpha=0.7)
    ax.bar(x + width/2, steered_to_pos - steered_to_neg, width, label='Steered', color='green', alpha=0.7)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_title("Distance Difference (closer to pos = negative)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("d(pos) - d(neg)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    base_closer = np.sum(base_to_pos < base_to_neg)
    steered_closer = np.sum(steered_to_pos < steered_to_neg)
    ax.text(0.02, 0.98, f"Closer to pos: Base {base_closer}/{len(base)}, Steered {steered_closer}/{len(steered)}",
            transform=ax.transAxes, fontsize=8, verticalalignment='top')
