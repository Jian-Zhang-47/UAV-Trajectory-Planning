import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from config import COVERAGE_RADIUS, NUM_UAVS
from scipy.spatial.distance import cdist

def merge_targets_dbscan(targets, eps=COVERAGE_RADIUS):
    clustering = DBSCAN(eps=eps, min_samples=1).fit(targets)
    merged = []
    cluster_centers = {}
    # Iterate over the unique cluster labels
    for label in np.unique(clustering.labels_):
        if label == -1:  # Noise points
            noise_points = targets[clustering.labels_ == -1]
            
            # Step 3: Find the nearest cluster center for each noise point
            for pt in noise_points:
                distances = [np.linalg.norm(pt - cluster_centers[cluster_label]) 
                             for cluster_label in cluster_centers]
                closest_cluster = np.argmin(distances)
                # Assign noise point to the nearest cluster center
                cluster_centers[closest_cluster].append(pt)
                
        else:  # Regular clusters
            cluster_points = targets[clustering.labels_ == label]
            # Compute the center (mean) of the cluster
            cluster_centers[label] = np.mean(cluster_points, axis=0)
            merged.append(cluster_centers[label])  # Add the cluster center
    
    # Step 4: Ensure all targets are covered
    all_covered = True
    for pt in targets:
        # Check if the target point is within the coverage radius of any cluster
        covered = False
        for cluster_center in cluster_centers.values():
            if np.linalg.norm(pt - cluster_center) <= eps:
                covered = True
                break
        if not covered:
            all_covered = False
            # Assign uncovered target to the nearest cluster center
            distances = [np.linalg.norm(pt - cluster_center) for cluster_center in cluster_centers.values()]
            closest_cluster = np.argmin(distances)
            merged.append(cluster_centers[closest_cluster])
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
