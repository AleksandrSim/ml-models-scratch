from torch.utils.data import TensorDataset, DataLoader

def data_merger(X, y):
    my_dataset = TensorDataset(X, y)
    my_dataloader = DataLoader(my_dataset, batch_size=4)

    return my_dataloader


