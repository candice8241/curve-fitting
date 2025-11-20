# -*- coding: utf-8 -*-
"""
Peak Clustering Module

Provides DBSCAN-based clustering for grouping nearby XRD peaks.

@author: candicewang928@gmail.com
"""

import numpy as np
from sklearn.cluster import DBSCAN


def cluster_peaks_dbscan(peak_positions, eps=None, min_samples=1):
    """
    Use DBSCAN density clustering to group nearby peaks.

    This function groups peaks that are close to each other in the 2theta space,
    which is useful for identifying overlapping peaks that should be fitted together.

    Parameters
    ----------
    peak_positions : array_like
        1D array of peak positions (e.g., 2theta values)
    eps : float, optional
        Maximum distance between two peaks to be in the same group.
        If None, automatically estimated as 1.5 times the median distance
        between adjacent peaks. Default is None.
    min_samples : int, optional
        Minimum number of peaks to form a cluster. Default is 1.

    Returns
    -------
    labels : ndarray
        Cluster labels for each peak. Initially, -1 indicates noise/outlier,
        but these are reassigned to nearest clusters.
    n_clusters : int
        Number of clusters found (after handling noise points)

    Notes
    -----
    Noise points (initially labeled -1) are automatically assigned to their
    nearest cluster to ensure all peaks are included in the analysis.

    Examples
    --------
    >>> peak_positions = np.array([10.5, 10.7, 15.2, 20.1, 20.3])
    >>> labels, n_clusters = cluster_peaks_dbscan(peak_positions, eps=1.0)
    >>> print(f"Found {n_clusters} groups")
    """
    if len(peak_positions) == 0:
        return np.array([]), 0

    if len(peak_positions) == 1:
        return np.array([0]), 1

    # Reshape for sklearn
    X = np.array(peak_positions).reshape(-1, 1)

    # Auto-estimate eps if not provided
    if eps is None:
        # Use median distance between adjacent peaks as eps
        sorted_pos = np.sort(peak_positions)
        if len(sorted_pos) > 1:
            distances = np.diff(sorted_pos)
            eps = np.median(distances) * 1.5  # 1.5x median distance
        else:
            eps = 1.0

    # Run DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = clustering.labels_

    # Handle noise points (label -1) by assigning them to nearest cluster
    noise_mask = labels == -1
    if np.any(noise_mask) and np.any(~noise_mask):
        for i in np.where(noise_mask)[0]:
            # Find nearest non-noise point
            non_noise_idx = np.where(~noise_mask)[0]
            distances = np.abs(peak_positions[non_noise_idx] - peak_positions[i])
            nearest = non_noise_idx[np.argmin(distances)]
            labels[i] = labels[nearest]
    elif np.all(noise_mask):
        # All points are noise, treat as single cluster
        labels = np.zeros(len(labels), dtype=int)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    return labels, n_clusters
