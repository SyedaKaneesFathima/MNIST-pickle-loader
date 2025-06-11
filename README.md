# MNIST Dataset Fetcher and Saver

This project demonstrates how to fetch the MNIST dataset using scikit-learn, split it into training and testing sets, save it as a pickle file (mnist.pkl), and later load it for future use.

## Project Structure

- Fetches MNIST data (mnist_784) from OpenML.
- Splits the dataset into:
  - 60,000 training samples
  - 10,000 testing samples
- Saves the dataset into a pickle file (mnist.pkl) in the current directory.
- Includes a helper function to load the saved dataset for further processing.

## Requirements

Make sure you have the following Python packages installed:

```bash
pip install numpy scikit-learn
