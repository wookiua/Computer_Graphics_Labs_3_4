import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import KMeans

def find_optimal_clusters(data, max_clusters=20):
    distortions = []

    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)

    plt.plot(range(1, max_clusters + 1), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()

    optimal_clusters = int(input('Enter the optimal number of clusters: '))
    return optimal_clusters

def main():

    data = np.loadtxt('DS5.txt')

    optimal_clusters = find_optimal_clusters(data)

    kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
    kmeans.fit(data)

    cluster_centers = kmeans.cluster_centers_

    vor = Voronoi(cluster_centers)

    fig, ax = plt.subplots(figsize=(960/80, 540/80))

    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=5)

    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue', line_width=1, line_alpha=0.3, point_size=0)

    ax.scatter(data[:, 0], data[:, 1], c='red', s=5, alpha=0.1)

    plt.savefig('result.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
