from manim import *
import numpy as np

class LloydsAlgorithm(Scene):
    def construct(self):
        # Parameters
        num_points = 30
        num_clusters = 3
        num_iterations = 4

        # Generate random points in 2D space
        points_x = np.random.rand(num_points) * 6 - 3  
        points_y = np.random.rand(num_points) * 6 - 3  
        

        # Create a group of dots at the random coordinates
        dots = VGroup(*[Dot(point=(coord[0], coord[1], 0)) for coord in np.column_stack((points_x, points_y))])

        # Create point dots in Manim
        # point_dots = VGroup(*[Dot(point=coords) for coords in points])
        self.add(dots)


def initialize_centroids(X, k):
    """Randomly initialize k centroids from the dataset."""
    indices = np.random.choice(len(X), k, replace=False)
    return X[indices]

def assign_clusters(X, centroids):
    """Assign each data point to the closest centroid."""
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    """Update centroid positions as the mean of assigned points."""
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids

def lloyds_algorithm(X, k, max_iters=100, tol=1e-4):
    """Performs K-means clustering using Lloyd's algorithm."""
    centroids = initialize_centroids(X, k)
    
    for i in range(max_iters):
        old_centroids = centroids
        labels = assign_clusters(X, centroids)
        centroids = update_centroids(X, labels, k)

        # Check for convergence
        if np.all(np.linalg.norm(centroids - old_centroids, axis=1) < tol):
            break
            
    return labels, centroids

