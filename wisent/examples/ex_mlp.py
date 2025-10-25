# Example of using MLPClassifier

# First we need to import the necessary modules
import numpy as np
import matplotlib.pyplot as plt
from wisent.classifiers.models.mlp import MLPClassifier
from wisent.classifiers.core.atoms import ClassifierTrainConfig
import torch

# Create some synthetic nonlinear data
def make_nonlinear_data(n=1000, noise=0.2, seed=24):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 2))
    y = ((X[:, 0]**2 + X[:, 1]**2) + rng.normal(scale=noise, size=n) > 0.5).astype(int)
    return X.astype(np.float32), y

# Generate data
X, y = make_nonlinear_data(n=1000, noise=0.2, seed=24)

# Plot the data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=20)
plt.title('Synthetic Nonlinear Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Class')
plt.show()

# Define training configuration, we are going to monitor accuracy as a metric
train_config = ClassifierTrainConfig(
    test_size=0.2,
    num_epochs=50,
    batch_size=32,
    learning_rate=1e-3,
    monitor='accuracy',
    random_state=42
)

# Initialize the MLPClassifier
classifier = MLPClassifier(
    threshold=0.5,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    hidden_dim=16
)

# Fit the classifier
report = classifier.fit(X, y, config=train_config)

print("\nTraining History on Non-Linear Data:")
for epoch in range(len(report.history['train_loss'])):
    print(f"Epoch {epoch+1}: Train Loss={report.history['train_loss'][epoch]:.4f}, "
          f"Val Loss={report.history['test_loss'][epoch]:.4f}, "
          f"Train Acc={report.history['accuracy'][epoch]:.4f}, "
    )

# Plot decision boundary

# Plot decision boundary for linear data and non-linear data
def plot_decision_boundary(clf: MLPClassifier, X: np.ndarray, y: np.ndarray, title: str):
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array(Z)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=20)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Class')
    plt.show()

plot_decision_boundary(classifier, X, y, "MLP Classifier Decision Boundary on Non-Linear Data")
