#!/usr/bin/env python3 
# 
# anchors.py  Andrew Belles  Feb 5th, 2026 
# 
# Computes centroid "anchors" across all channels of image dataset for use in 
# SpatialGAT model. Anchors are used as prototypical node types to induce heterogeneity, 
# persuant to Jianwen et. al 
# 

import argparse, torch 

import numpy as np 

from sklearn.cluster       import KMeans 

from torch.utils.data      import DataLoader 

from preprocessing.loaders import load_spatial_mmap_manifest 


def viirs_patch_features(x, patch_size=32): 

    B, C, _, _ = x.shape
    P = patch_size 

    patches = x.unfold(2, P, P).unfold(3, P, P)
    Gh, Gw  = patches.shape[2], patches.shape[3]
    L       = Gh * Gw 
    patches = patches.contiguous().view(B, C, L, P, P).permute(0, 2, 1, 3, 4)
    patches = patches.reshape(B * L, C, P, P)

    flat    = patches.view(patches.size(0), C, -1)
    p95     = torch.quantile(flat, 0.95, dim=2)
    return p95[:, :2]

def usps_patch_features(x, patch_size=32): 

    B, C, _, _ = x.shape
    P = patch_size 

    patches = x.unfold(2, P, P).unfold(3, P, P)
    Gh, Gw  = patches.shape[2], patches.shape[3]
    L       = Gh * Gw 
    patches = patches.contiguous().view(B, C, L, P, P).permute(0, 2, 1, 3, 4)
    patches = patches.reshape(B * L, C, P, P)

    flat = patches.view(patches.size(0), C, -1)

    p95 = torch.quantile(flat, 0.95, dim=2)

    return p95 

def compute_anchors(
    loader, 
    n_samples=5e5, 
    k=3, 
    patch_size=32, 
    device="cuda", 
    feature_fn=None,
    min_norm=1e-6
): 
    if feature_fn is None: 
        raise ValueError("feature_fn is required.")

    max_samples = n_samples * 2 
    buffer      = []
    total_valid = 0
    total_raw   = 0 

    for batch in loader: 
        xb = batch[0] if isinstance(batch, (list, tuple)) else batch 
        xb = xb.to(device)

        feats      = (feature_fn(xb, patch_size=patch_size)
                      .detach()
                      .cpu()
                      .numpy()
                      .astype(np.float32, copy=False)) 
        
        total_raw += feats.shape[0]
        norms      = np.linalg.norm(feats, axis=1)
        valid      = np.isfinite(norms) & (norms > float(min_norm))
        if valid.any():
            keep   = feats[valid]
            buffer.append(keep)
            total_valid += keep.shape[0]

        if total_valid >= max_samples: 
            break 

    if not buffer: 
        raise ValueError("No valid anchor features after zero filtering")

    data    = np.vstack(buffer)
    
    # normalization to z-score 
    mean = data.mean(axis=0)
    std  = data.std(axis=0)
    std  = np.maximum(std, 1e-6)

    z    = (data - mean) / std 

    take    = min(int(n_samples), z.shape[0])
    idx     = np.random.choice(z.shape[0], take, replace=False)
    samples = z[idx]

    kmeans  = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(samples)
    anchors = kmeans.cluster_centers_.astype(np.float32)

    order   = np.argsort(np.linalg.norm(anchors, axis=1))
    anchors = anchors[order]

    keep_ratio = float(total_valid) / max(total_raw, 1)
    print(f"[anchors] kept {total_valid}/{total_raw} patches ({keep_ratio:.2%})")

    return anchors, np.array([mean, std], dtype=np.float32)


def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["viirs", "usps"], required=True)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-samples", type=int, default=500_000) 
    parser.add_argument("--min-norm", type=float, default=1e-6)
    args = parser.parse_args()

    if args.mode == "viirs": 
        feature_fn  = viirs_patch_features 
        in_channels = 2 
    else: 
        feature_fn = usps_patch_features 
        in_channels = 4 

    bags    = load_spatial_mmap_manifest(args.root, tile_shape=(in_channels, 256, 256), max_bag_size=64)
    dataset = bags["dataset"]
    collate = bags["collate_fn"]

    loader  = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate 
    )

    anchors, stats = compute_anchors(
        loader, 
        n_samples=args.n_samples, 
        k=3,
        patch_size=args.patch_size,
        feature_fn=feature_fn,
        min_norm=args.min_norm
    )

    np.save(args.out, anchors)

    stats_path = str(args.out + "_stats.npy")
    np.save(stats_path, stats)

    print(f"[anchors] saved to {args.out}, shape={anchors.shape}")
    print(anchors)
    print(stats)

if __name__ == "__main__": 
    main() 
