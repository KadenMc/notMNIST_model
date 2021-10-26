from PIL import Image # Only used to check image validity

# NumPy & PyTorch imports
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms



class MyDataset(Dataset):
    """
    A custom PyTorch Dataset class to handle data conversion
    and transforms.
    """
    def __init__(self, data, targets,  transform=False):
        if data.dtype != torch.float32:
            self.data = torch.FloatTensor(data)
        else:
            self.data = data 
        
        if targets.dtype != torch.int64:
            self.targets = torch.LongTensor(targets)
        else:
            self.targets = targets
        
        self.classes = np.sort(np.unique(self.targets))
        self.transform = transform
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        if self.transform:
          x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.data)


def is_valid_file(path):
    """
    Checks whether an image is valid, or corrupt.
    """
    try:
        img = Image.open(path)
        img.verify()
    except:
        return False

    return True


def load_apply_transforms(dataset):
    """
    PyTorch applies loads data and applies transforms only when called,
    not when defined. This means that we'll be repeating the image loading
    and transformations on each batch iteration. We can avoid this by
    loading and applying the transformations to all the data now.
    
    Note that this is only possible when there is memory for all of the data.
    """
    dataloader = DataLoader(dataset, batch_size=len(dataset))
    X, y = next(iter(dataloader))
    return MyDataset(X, y)


def prepare_dataloaders(data_path, train_n, val_n, load_in_batches, batch_size):
    # Define the data transform
    transform = transforms.Compose([
        transforms.Grayscale(), # Convert to grayscale
        transforms.ToTensor(), # Make into PyTorch tensor
        transforms.Normalize((0,), (1,)), # Normalization
    ])
    
    # Define dataset using ImageFolder
    dataset = datasets.ImageFolder(root=data_path, transform=transform, is_valid_file=is_valid_file)

    if not load_in_batches:
        # If we're not loading in batches, we load all the data at once
        # and we can then insert it into a new, loaded dataset
        # The advantage of this is speed: We don't need to keep loading images from file
        # But it may not be feasible if the dataset is too large
        dataset = load_apply_transforms(dataset)
    
    # Split into train, validation, and testing sets
    # For the purposes of reproducibility, set the seed to always
    # get the same training, validation, and testing sets
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, \
        [train_n, val_n, len(dataset) - (train_n + val_n)], \
        generator=torch.Generator().manual_seed(1005210558))

    # Create and return data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, test_loader