import numpy as np
import matplotlib.pyplot as plt

class MarkovCalculations: 
    def __init__(self, transition_matrix) -> None:
        if not np.allclose(transition_matrix.sum(axis=1), 1):
            raise ValueError("Each row of the transition matrix must sum to 1.")
        self.transition_matrix = transition_matrix

    def calculate_stationary_distribution(self):
        num_states = self.transition_matrix.shape[0]
        A = self.transition_matrix.T - np.eye(num_states)
        A = np.vstack([A, np.ones(num_states)])  # Add the constraint sum = 1
        b = np.zeros(num_states + 1)
        b[-1] = 1 # solve for states  
        stationary_distribution = np.linalg.lstsq(A, b, rcond=None)[0]
        return stationary_distribution

    def run_markov_chain(self, initial_state, num_steps, true_distribution):
        np.random.seed(55)  # Seed is 55 in the slide images
        num_states = self.transition_matrix.shape[0]
        if initial_state < 0 or initial_state >= num_states:
            raise ValueError("Initial state must be a valid index in the transition matrix.")
        current_state = initial_state
        path = [current_state]

        # Track cumulative state frequencies over time
        cumulative_counts = np.zeros(num_states)

        # Run the Markov chain
        differences = []
        for step in range(1, num_steps + 1):
            current_state = np.random.choice(
                num_states, p=self.transition_matrix[current_state]
            )
            path.append(current_state)
            cumulative_counts[current_state] += 1

            # Calculate frequencies at this step
            frequencies = cumulative_counts / step

            # Compute the difference from the true distribution
            differences.append(np.abs(frequencies - true_distribution).mean())

        results = {
            "final_state_frequencies": cumulative_counts / num_steps,
            "path": path,
            "differences": differences
        }
        return results

if __name__ == "__main__":
    transition_matrix = np.array([
        [0.5, 0.3, 0.2], 
        [0.2, 0.2, 0.6],
        [0.4, 0.4, 0.2]
    ])
    # transition_matrix = np.array([[0.7, 0.2, 0.1], 
    #                               [0.3, 0.6, 0.1], 
    #                               [0.3, 0.2, 0.5]])  # Used to tetst code, example from class 
    c = MarkovCalculations(transition_matrix=transition_matrix)
    true_distribution = c.calculate_stationary_distribution()
    print("True Stationary Distribution:")
    for i, prob in enumerate(true_distribution):
        print(f"State {i}: {prob:.4f}")

    initial_state = 0  
    num_steps = 5000
    results = c.run_markov_chain(initial_state, num_steps, true_distribution)

    # Plotting the differences over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_steps + 1), results["differences"], label="Simulating")
    plt.xlabel("Number of Steps")
    plt.ylabel("Mean Absolute Difference")
    plt.title("Convergence to True Stationary Distribution (Mean Absolute Difference)")
    plt.legend()
    plt.grid()
    plt.show()

    # Print final state frequencies
    print("\nFinal State Frequencies:")
    for state, freq in enumerate(results["final_state_frequencies"]):
        print(f"Simulated: State {state}: {freq:.4f}")
    
    for state, freq in enumerate(c.calculate_stationary_distribution()):
        print(f"Real: State {state}: {freq:.4f}")
