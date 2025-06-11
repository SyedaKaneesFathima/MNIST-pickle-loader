import numpy as np
from sklearn.datasets import fetch_openml
import pickle
import os

# Get the directory of the current Python file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Full path where we want to save the mnist.pkl
save_path = os.path.join(current_dir, "mnist.pkl")

# Fetch MNIST data directly
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]

# Convert labels to integer
y = y.astype(np.uint8)

# Split into training and test sets (60k train, 10k test)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# Save data into pickle file
mnist_dict = {
    "training_images": X_train,
    "training_labels": y_train,
    "test_images": X_test,
    "test_labels": y_test
}

with open(save_path, 'wb') as f:
    pickle.dump(mnist_dict, f)

print("Save complete.")
print(f"mnist.pkl saved at: {save_path}")

# Example to load the data later:
def load():
    with open(save_path, 'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

# Testing load function
if __name__ == '__main__':
    X_train_loaded, y_train_loaded, X_test_loaded, y_test_loaded = load()
    print("Loaded training images shape:", X_train_loaded.shape)
    print("Loaded test images shape:", X_test_loaded.shape)