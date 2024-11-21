from manim import *
import numpy as np
from PIL import Image

class LloydsAlgorithm(Scene):
    def construct(self):
        np.random.seed(55) # Seed is 55 in the slide images

        # Parameters
        num_points = 30
        num_clusters = 3
        num_iterations = 4

        # Generate random points in 2D space
        points_x = np.random.rand(num_points) * 6 - 3
        points_y = np.random.rand(num_points) * 6 - 3
        points = np.column_stack((points_x, points_y))

        # Create a group of dots at the random coordinates
        dots = VGroup(*[Dot(point=(coord[0], coord[1], 0), color=WHITE) for coord in points])
        self.add(dots)

        # Initialize centroids and create centroid dots
        centroids = initialize_centroids(points, num_clusters)
        centroid_dots = VGroup(*[Dot(point=(coord[0], coord[1], 0), color=YELLOW).scale(1.2) for coord in centroids])
        self.add(centroid_dots)

        for iteration_num in range(num_iterations):
            # Assign points to nearest centroids and color them
            labels = assign_clusters(points, centroids)
            for i, dot in enumerate(dots):
                dot.set_color([RED, BLUE, GREEN][labels[i]])  # Colors for each cluster
            
            # Update centroid positions and animate the movement of centroids
            new_centroids = update_centroids(points, labels, num_clusters)
            self.play(*[centroid_dot.animate.move_to((new_position[0], new_position[1], 0)) for centroid_dot, new_position in zip(centroid_dots, new_centroids)], run_time=1)
            # Update centroids for the next iteration
            frame_data = self.camera.get_image()
            frame_data.save(f"media/images/scribe3/frame_iteration_{iteration_num}.png")
            centroids = new_centroids
            self.wait(0.5)


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
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])
