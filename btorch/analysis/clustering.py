import numpy as np
from fastdtw import fastdtw
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import euclidean


def suggest_threshold(linkage_matrix):
    """Suggest a sensible threshold for hierarchical clustering."""
    sorted_distances = sorted(linkage_matrix[:, 2], reverse=True)
    diffs = np.diff(sorted_distances)
    potential_elbows = np.where(diffs > np.mean(diffs))[0] + 1

    if len(potential_elbows) == 0:
        suggested_threshold = sorted_distances[-1]
    else:
        first_elbow = potential_elbows[0]
        suggested_threshold = sorted_distances[first_elbow]

    return suggested_threshold


def cluster_traces(traces, threshold=10, linkage_method="average"):
    """Cluster traces of voltages or currents using hierarchical agglomerative
    clustering with DTW distance."""
    num_traces = len(traces)

    distance_matrix = np.zeros((num_traces, num_traces))
    for i in range(num_traces):
        for j in range(num_traces):
            distance, _ = fastdtw(traces[i], traces[j], dist=euclidean)
            distance_matrix[i, j] = distance

    Z = linkage(distance_matrix, method=linkage_method)
    clusters = fcluster(Z, t=threshold, criterion="distance")

    cluster_indices = {}
    for idx, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_indices:
            cluster_indices[cluster_id] = []
        cluster_indices[cluster_id].append(idx)

    return cluster_indices, clusters, Z, distance_matrix
