import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset


def load_data(dataset: str = "mnist"):
    if dataset == "mnist":
        # Load train/test data
        data_train = pd.read_csv("datasets/MNIST/mnist_train.csv")
        data_test = pd.read_csv("datasets/MNIST/mnist_test.csv")

        # Separate labels and image pixel values - and convert to numpy arrays
        y_train = data_train["label"].values
        x_train = data_train.drop(["label"], axis=1).values
        y_test = data_test["label"].values
        x_test = data_test.drop(["label"], axis=1).values

        # Normalize pixel values to be between 0 and 1
        x_train = x_train / 255
        x_test = x_test / 255

        # Reshape data to be in format [samples height width] (grayscale images only have one channel)
        # For CNNs, we would reshape to [samples channels height width] w/ .reshape(-1, 1, 28, 28)
        # x_train = x_train.reshape(-1, 28, 28)
        # x_test = x_test.reshape(-1, 28, 28)

        # Convert to torch tensors
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.int64)
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.int64)

        # Create train/test datasets
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)

        return train_dataset, test_dataset
    else:
        raise NotImplementedError(
            f"Dataset {dataset} not implemented yet. Please choose from ['mnist']"
        )
