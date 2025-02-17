from __future__ import print_function

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader # Import DataLoader and Dataset from PyTorch

from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader # Import DataLoader and Dataset from PyTorch
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from customloader import *



class ImprovedClassifier(nn.Module):
    def __init__(self):
        super(ImprovedClassifier, self).__init__()
        self.fc1 = nn.Linear(7, 128)  # Input dimension adjusted to X_train's feature count
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 4)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, file_path, transform=None):
        # Step 1: Load the data
        data = pd.read_csv(file_path)

        # Separate features (X) and labels (y)
        self.X = data.iloc[:, :-1].values  # Features
        self.targets = data.iloc[:, -1].values  # Labels

        # Step 2: Apply SMOTE to balance the classes
        smote = SMOTE(sampling_strategy={0: 250, 1: 250, 2: 250, 3: 250}, random_state=42)
        self.X, self.targets = smote.fit_resample(self.X, self.targets)

        # Check the distribution of the new dataset
        print(pd.Series(self.targets).value_counts())  # Should print 250 for each class

        # Step 3: Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.targets, test_size=0.2, random_state=42,
                                                          stratify=self.targets)

        # Step 4: Standardize the features using StandardScaler
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_val = self.scaler.transform(X_val)

        # Step 5: Store the labels
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.y_val = torch.tensor(y_val, dtype=torch.long)

        # Optional transformation
        self.transform = transform

    def __len__(self):
        # Return the total number of samples (training + validation)
        return len(self.X_train) + len(self.X_val)

    def __getitem__(self, idx):
        # If idx is in the training set
        if idx < len(self.X_train):
            sample = torch.tensor(self.X_train[idx], dtype=torch.float32)
            targets = self.y_train[idx]
        else:
            idx -= len(self.X_train)  # Adjust index for validation set
            sample = torch.tensor(self.X_val[idx], dtype=torch.float32)
            targets = self.y_val[idx]

        # Apply transformations if any are provided
        if self.transform:
            sample = self.transform(sample)

        return sample, targets


def getDataset():
    file_path = '/content/drive/MyDrive/ICDCS/modified_rock_data.csv'  # Path to your dataset
    dataset = CustomDataset(file_path)
    return  dataset





def basic_loader(num_clients, loader_type):
    dataset = getDataset()
    #print("Here the legnth of mydataset ",len(dataset))
    return loader_type(num_clients, dataset)


def train_dataloader(num_clients, loader_type='iid', store=True, path='./data/loader.pk'):
    assert loader_type in ['iid', 'byLabel', 'dirichlet'],\
        'Loader has to be either \'iid\' or \'non_overlap_label \''
    if loader_type == 'iid':
        loader_type = iidLoader
    elif loader_type == 'byLabel':
        loader_type = byLabelLoader
    elif loader_type == 'dirichlet':
        loader_type = dirichletLoader

    if store:
        try:
            with open(path, 'rb') as handle:
                loader = pickle.load(handle)
        except:
            print('Loader not found, initializing one')
            loader = basic_loader(num_clients, loader_type)
            print('Save the dataloader {}'.format(path))
            with open(path, 'wb') as handle:
                pickle.dump(loader, handle)
    else:
        print('Initialize a data loader')
        loader = basic_loader(num_clients, loader_type)
    
    print("Loader length ",len(loader))
    return loader








# Create a separate Dataset for test data (from validation)
class TestDataset(Dataset):
    def __init__(self, X_val, y_val, transform=None):
        """
        Initializes the test dataset.

        :param X_val: The validation features (to be used as test data).
        :param y_val: The validation labels (to be used as test labels).
        :param transform: Optional transformation to apply to the data.
        """
        self.X_test = X_val
        self.y_test = y_val
        self.transform = transform

    def __len__(self):
        return len(self.X_test)

    def __getitem__(self, idx):
        sample = torch.tensor(self.X_test[idx], dtype=torch.float32)
        targets = self.y_test[idx]

        # Apply transformations if any are provided
        if self.transform:
            sample = self.transform(sample)

        return sample, targets
    





class CustomTestDataset(Dataset):
    def __init__(self,file_path, transform=None):
        """
        Initializes the dataset for test/validation data.

        :param file_path: Path to the data file (CSV).
        :param transform: Optional transformation to apply to the data.
        """
        # Step 1: Load the data
        data = pd.read_csv(file_path)

        # Separate features (X) and labels (y)
        self.X = data.iloc[:, :-1].values  # Features
        self.targets = data.iloc[:, -1].values  # Labels

        # Step 2: Apply SMOTE to balance the classes (SMOTE can be applied to test data in some cases)
        smote = SMOTE(sampling_strategy={0: 250, 1: 250, 2: 250, 3: 250}, random_state=42)
        self.X, self.targets = smote.fit_resample(self.X, self.targets)

        # Step 3: Split the data into training and validation sets (test data here will be `X_val`)
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.targets, test_size=0.2, random_state=42,
                                                          stratify=self.targets)

        # Step 4: Standardize the features using StandardScaler
        self.scaler = StandardScaler()
        self.X_val = self.scaler.fit_transform(X_val)  # Only standardize validation/test data
        self.y_val = torch.tensor(y_val, dtype=torch.long)

        # Optional transformation
        self.transform = transform

    def __len__(self):
        # Return the total number of samples in the validation set (test set)
        return len(self.X_val)

    def __getitem__(self, idx):
        # Fetch the sample and label based on index from the validation set
        sample = torch.tensor(self.X_val[idx], dtype=torch.float32)
        targets = self.y_val[idx]

        # Apply transformations if any are provided
        if self.transform:
            sample = self.transform(sample)

        return sample, targets











def test_dataloader(test_batch_size):
    file_path = '/content/drive/MyDrive/ICDCS/modified_rock_data.csv'
    test_loader = torch.utils.data.DataLoader(CustomTestDataset(file_path), batch_size=test_batch_size)
    #print(len(test_loader))
    return test_loader


if __name__ == '__main__':
    from torchsummary import summary

    print("#Initialize a network")
    net = ImprovedClassifier()
    summary(net.cuda(), (1, 32, 32))

    print("\n#Initialize dataloaders")
    loader_types = ['iid', 'byLabel', 'dirichlet']
    for i in range(len(loader_types)):
        loader = train_dataloader(10, loader_types[i], store=False)
        print(f"Initialized {len(loader)} loaders (type: {loader_types[i]}), each with batch size {loader.bsz}.\
        \nThe size of dataset in each loader are:")
        print([len(loader[i].dataset) for i in range(len(loader))])
        print(f"Total number of data: {sum([len(loader[i].dataset) for i in range(len(loader))])}")

    print("\n#Feeding data to network")
    x = next(iter(loader[i]))[0].cuda()
    y = net(x)
    print(f"Size of input:  {x.shape} \nSize of output: {y.shape}")
