import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from moabb.datasets import BNCI2014_004
from moabb.paradigms import LeftRightImagery
from scipy.linalg import fractional_matrix_power


def euclidean_alignment(X):
    """Computes Euclidean Alignment whitening transform per subject."""
    covs = [np.dot(x, x.T) / x.shape[1] for x in X]
    mean_cov = np.mean(covs, axis=0)
    W = fractional_matrix_power(mean_cov, -0.5).real
    X_aligned = np.zeros_like(X)
    for i in range(X.shape[0]):
        X_aligned[i] = np.dot(W, X[i])
    return X_aligned


def extract_spatial_features(X):
    """
    Extract spatial covariance features for each trial [trials, channels, time].
    Returns upper-triangle feature representation per trial.
    """
    n_trials, n_ch, _ = X.shape
    features = []
    triu_idx = np.triu_indices(n_ch)
    for i in range(n_trials):
        cov = np.dot(X[i], X[i].T) / X[i].shape[1]
        features.append(cov[triu_idx])
    return np.array(features)


def main():
    os.makedirs('results', exist_ok=True)
    print("Generating pure t-SNE visualization (Before vs After Euclidean Alignment)...")

    dataset = BNCI2014_004()
    paradigm = LeftRightImagery()

    subjects_to_plot = list(range(1, 6))
    raw_feats_list = []
    aligned_feats_list = []
    subject_labels_list = []

    for sub in subjects_to_plot:
        X_sub, _, _ = paradigm.get_data(dataset=dataset, subjects=[sub])
        X_sub = X_sub[:, :, :1000]
        
        # Raw trial features
        raw_feat = extract_spatial_features(X_sub)
        raw_feats_list.append(raw_feat)
        
        # Aligned trial features
        X_aligned = euclidean_alignment(X_sub)
        aligned_feat = extract_spatial_features(X_aligned)
        aligned_feats_list.append(aligned_feat)
        
        subject_labels_list.extend([sub] * len(X_sub))

    X_raw_feats = np.concatenate(raw_feats_list, axis=0)
    X_aligned_feats = np.concatenate(aligned_feats_list, axis=0)
    sub_labels = np.array(subject_labels_list)

    print("Computing t-SNE projections on spatial covariance features...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    
    tsne_raw = tsne.fit_transform(X_raw_feats)
    tsne_aligned = tsne.fit_transform(X_aligned_feats)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for idx, sub_id in enumerate(subjects_to_plot):
        mask = (sub_labels == sub_id)
        
        axes[0].scatter(tsne_raw[mask, 0], tsne_raw[mask, 1], 
                        c=colors[idx], label=f'Subject {sub_id}', alpha=0.7, s=25)
        
        axes[1].scatter(tsne_aligned[mask, 0], tsne_aligned[mask, 1], 
                        c=colors[idx], label=f'Subject {sub_id}', alpha=0.7, s=25)

    axes[0].set_title('Before Euclidean Alignment (Raw t-SNE)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('t-SNE Dimension 1', fontsize=11)
    axes[0].set_ylabel('t-SNE Dimension 2', fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('After Euclidean Alignment (Whitened t-SNE)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('t-SNE Dimension 1', fontsize=11)
    axes[1].set_ylabel('t-SNE Dimension 2', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(title='Subject ID', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig('results/tsne_euclidean_alignment.png', dpi=300, bbox_inches='tight')
    print("✅ Pure t-SNE plot saved successfully to 'results/tsne_euclidean_alignment.png'!")


if __name__ == '__main__':
    main()