from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Training data: [p1 strength, p1 health, p2 strength, p2 health], labels: 0 (p2 won), 1 (p1 win)
X = [
    [110, 10, 400, 10],
    [23, 42, 32, 20],
    [20, 20, 19, 21],
    [20, 21, 32, 3],
]
y = [0, 0, 1, 0]

# Create the decision tree classifier
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)

# Visualize the decision tree
plt.figure(figsize=(10, 6))
tree.plot_tree(
    clf,
    feature_names=["P1 Strength", "P1 Health", "P2 Strength", "P2 Health"],
    class_names=["P2 Win", "P1 Win"],
    filled=True,
    proportion=False,  # Avoid scaling boxes by the number of samples
    rounded=True,      # Add rounded corners for better aesthetics
    fontsize=10        # Adjust font size for better readability
)
plt.show()

