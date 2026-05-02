import numpy as np
import time
from sklearn.cluster import KMeans
import argparse

def gaussian_density(X, width, height):
    # Center at (width/2, height/2), sigma = 0.25 * min(width, height)
    cx, cy = width / 2.0, height / 2.0
    sigma = 0.25 * min(width, height)
    dx = X[:, 0] - cx
    dy = X[:, 1] - cy
    dist_sq = dx**2 + dy**2
    return np.exp(-dist_sq / (2.0 * sigma**2))

def main():
    parser = argparse.ArgumentParser(description="K-Means Benchmark vs Voronoi CVT")
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--sites", type=int, default=256)
    parser.add_argument("--iter", type=int, default=10)
    args = parser.parse_args()

    print(f"[Python K-Means] grid={args.width}x{args.height}  clusters={args.sites}  max_iter={args.iter}")
    
    # 1. Generate discrete grid spanning the width and height
    x = np.arange(args.width)
    y = np.arange(args.height)
    xv, yv = np.meshgrid(x, y)
    
    # X array shape: (N, 2)
    X = np.column_stack((xv.ravel(), yv.ravel()))
    
    # 2. Compute weights (using Gaussian density equivalent to C++ code)
    print("Generating density weights...")
    weights = gaussian_density(X, args.width, args.height)
    
    # 3. Setup Scikit-learn K-Means
    # Using 'lloyd' to match exact Lloyd's relaxation used in C++ and 'random' to match uniform inverse sampling.
    kmeans = KMeans(
        n_clusters=args.sites,
        init='random',
        n_init=1,
        max_iter=args.iter,
        tol=0.0,
        algorithm='lloyd'
    )
    
    # 4. Benchmark Fit
    print(f"Running sklearn.cluster.KMeans on {X.shape[0]} points...")
    start_time = time.time()
    
    kmeans.fit(X, sample_weight=weights)
    
    elapsed_ms = (time.time() - start_time) * 1000.0
    
    print("-" * 50)
    print(f"Python KMeans Time: {elapsed_ms:.1f} ms")
    print(f"Avg Time per iter: {elapsed_ms / kmeans.n_iter_:.1f} ms")
    print(f"Iterations run: {kmeans.n_iter_}")
    print("-" * 50)

if __name__ == "__main__":
    main()
