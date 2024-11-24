import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def detect_fire_clusters(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image")
    else:
        print("Image loaded successfully by OpenCV")

    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for red color (fire points)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for red regions
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    # Find coordinates of fire points
    fire_points = np.column_stack(np.where(mask > 0))
    
    if len(fire_points) == 0:
        raise ValueError("No fire points detected")
    else:
        print("fire points detected:"+str(len(fire_points)))
    # Cluster the fire points using DBSCAN
    clustering = DBSCAN(eps=30, min_samples=5).fit(fire_points)
    
    # Get unique clusters
    n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    
    # Create figure with two subplots for verification
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot original image with clusters
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image with Clusters')
    
    # Plot clusters with different colors
    colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
    for i, color in enumerate(colors):
        cluster_points = fire_points[clustering.labels_ == i]
        ax1.scatter(cluster_points[:, 1], cluster_points[:, 0], 
                   color=color, alpha=0.6, s=50)
    
    # Plot mask for verification
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('Fire Detection Mask')
    
    # Add cluster statistics
    cluster_sizes = []
    for i in range(n_clusters):
        cluster_size = np.sum(clustering.labels_ == i)
        cluster_sizes.append(cluster_size)
    
    plt.suptitle(f'Detected Fire Clusters: {n_clusters}\n'
                 f'Total Fire Points: {len(fire_points)}\n'
                 f'Average Cluster Size: {np.mean(cluster_sizes):.1f} points')
    
    # Remove axes for cleaner visualization
    ax1.axis('off')
    ax2.axis('off')
    
    # Ensure tight layout
    plt.tight_layout()
    
    return {
        'total_clusters': n_clusters,
        'cluster_sizes': cluster_sizes,
        'total_fire_points': len(fire_points),
        'visualization': plt,
        'mask': mask,
        'labeled_clusters': clustering.labels_
    }

def verify_overlay(result):
    """
    Print verification statistics about the clustering
    """
    print(f"Verification Results:")
    print(f"- Total number of clusters: {result['total_clusters']}")
    print(f"- Total fire points detected: {result['total_fire_points']}")
    print(f"- Cluster sizes: {result['cluster_sizes']}")
    print(f"- Average cluster size: {np.mean(result['cluster_sizes']):.1f}")
    print(f"- Largest cluster size: {max(result['cluster_sizes'])}")
    print(f"- Smallest cluster size: {min(result['cluster_sizes'])}")

# Example usage
result = detect_fire_clusters('fire_points_northern_india.jpg')
verify_overlay(result)
result['visualization'].show()
