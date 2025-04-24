import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from config import COVERAGE_RADIUS, NUM_UAVS
from scipy.spatial.distance import cdist

def merge_targets_dbscan(targets, eps=COVERAGE_RADIUS):
    clustering = DBSCAN(eps=eps, min_samples=1).fit(targets)
    merged = []
    for label in np.unique(clustering.labels_):
        cluster_points = targets[clustering.labels_ == label]
        merged.append(np.mean(cluster_points, axis=0))
    return np.array(merged)

def assign_targets_kmeans(uavs, targets, num_uavs=NUM_UAVS):
    kmeans = KMeans(n_clusters=num_uavs, init=uavs, n_init=1, random_state=42)
    labels = kmeans.fit_predict(targets)
    assigned = {i: [] for i in range(num_uavs)}
    for label, target in zip(labels, targets):
        assigned[label].append(target)
    for i in assigned:
        assigned[i] = np.array(assigned[i])
    return assigned
