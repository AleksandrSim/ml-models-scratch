import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from starter3 import *

# Now it is time to convert the data into tensors in order to train our model
def convert_data_to_tensor(dataset, input_size):
    # Declaring a features array
    # for each one of the dimensions, populate the features≠ array with features from the training set
    features = []
    for row in dataset:
        features.append(row[1])
    # Now we have to generate tensors for these features
    feature_tensors = []
    for feature in features:
        list_of_features = []
        for column in feature:
            list_of_features.append(int(column))
        feature_tensors.append(np.array(list_of_features))
    feature_tensors = np.array(feature_tensors)
    feature_tensors_reshaped = feature_tensors.reshape(-1, input_size)

    feature_tensors_reshaped = torch.tensor(feature_tensors_reshaped)
    # print(type(feature_tensors_reshaped[0][0]))
    feature_tensors_reshaped = feature_tensors_reshaped.type(torch.FloatTensor)

    return feature_tensors_reshaped


# For Q1 and Q4

def tensor_list(attributes):
    list_of_tensors = []

    for row in attributes:
        list_of_tensors.append(torch.tensor(row))

    return list_of_tensors


# create a list of tensors from a list of labels
def tensor_labels(labels):
    list_of_tensors = []

    for label in labels:
        temp = [0, 0, 0]
        temp[label] = 1
        list_of_tensors.append(torch.tensor(temp, dtype=torch.float))

    return torch.stack(list_of_tensors)

class InsurabilityDataset(torch.utils.data.Dataset):
    def __init__(self, file_string):
        data = read_insurability(file_string)
        unprocessed_labels, unprocessed_attributes = seperate_insurability(data)
        self.attributes = torch.tensor(unprocessed_attributes)
        self.labels = tensor_labels(unprocessed_labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.index_select(self.attributes, 0, torch.tensor(idx)).squeeze(-2), torch.index_select(self.labels, 0,
                                                                                                         torch.tensor(
                                                                                                             idx)).squeeze(
            -2)
