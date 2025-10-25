# Example of using LogisticClassifier

# First we need to import the necessary modules
import torch
from wisent.classifiers.models.logistic import LogisticClassifier
from wisent.classifiers.core.atoms import ClassifierTrainConfig
import numpy as np
import matplotlib.pyplot as plt

# Create some synthetic data
def make_linear_data(n=1000, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 2))
    y = (X[:, 0] + X[:, 1] + rng.normal(scale=noise, size=n) > 0).astype(int)
    return X.astype(np.float32), y

# Generate data
X, y = make_linear_data(n=1000, noise=0.1, seed=42)

# Plot the data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=20)
plt.title('Synthetic Linear Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Class')
plt.show()

# Define training configuration, we are going to monitor accuracy as a metric
train_config = ClassifierTrainConfig(
    test_size=0.2,
    num_epochs=20,
    batch_size=32,
    learning_rate=1e-2,
    monitor='accuracy',
    random_state=42
)

# Initialize the LogisticClassifier
classifier = LogisticClassifier(
    threshold=0.5,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Fit the classifier
report = classifier.fit(X, y, config=train_config)

# Print the training report

# Print history
print("Training History:")
for epoch in range(len(report.history['train_loss'])):
    print(f"Epoch {epoch+1}: Train Loss={report.history['train_loss'][epoch]:.4f}, "
          f"Val Loss={report.history['test_loss'][epoch]:.4f}, "
          f"Train Acc={report.history['accuracy'][epoch]:.4f}, "
    )

# Now create a non-linear dataset and see that LogisticClassifier struggles
def make_nonlinear_data(n=1000, noise=0.2, seed=24):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 2))
    y = ((X[:, 0]**2 + X[:, 1]**2) + rng.normal(scale=noise, size=n) > 0.5).astype(int)
    return X.astype(np.float32), y

# Generate non-linear data
X_nl, y_nl = make_nonlinear_data(n=1000, noise=0.2, seed=24)

# Plot the non-linear data
plt.figure(figsize=(8, 6))
plt.scatter(X_nl[:, 0], X_nl[:, 1], c=y_nl, cmap='coolwarm', edgecolor='k', s=20)
plt.title('Synthetic Non-Linear Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Class')
plt.show()

# create a new classifier instance for non-linear data
classifier_nl = LogisticClassifier(
    threshold=0.5,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Fit the classifier on non-linear data
report_nl = classifier_nl.fit(X_nl, y_nl, config=train_config)

# Print history for non-linear data
print("\nTraining History on Non-Linear Data:")
for epoch in range(len(report_nl.history['train_loss'])):
    print(f"Epoch {epoch+1}: Train Loss={report_nl.history['train_loss'][epoch]:.4f}, "
          f"Val Loss={report_nl.history['test_loss'][epoch]:.4f}, "
          f"Train Acc={report_nl.history['accuracy'][epoch]:.4f}, "
    )

# Plot decision boundary for linear data and non-linear data
def plot_decision_boundary(clf: LogisticClassifier, X: np.ndarray, y: np.ndarray, title: str):
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

plot_decision_boundary(classifier, X, y, 'Decision Boundary on Linear Data')
plot_decision_boundary(classifier_nl, X_nl, y_nl, 'Decision Boundary on Non-Linear Data')
