import numpy as np

import torch

def create_dataloader(dataset, batch_size, shuffle=True, num_workers=1, collate_fn=None):
    """Create a dataloader for the given dataset
    
    Arguments:
        dataset {torch.utils.data.Dataset} -- Dataset
        batch_size {int} -- Batch size
    
    Keyword Arguments:
        shuffle {bool} -- Whether to shuffle the dataset or not (default: {True})
        num_workers {int} -- Number of workers (default: {1})
        collate_fn -- Custom function to preprocess batch (e.g. padding sequences to same length).
    
    Returns:
        torch.utils.data.DataLoader -- Data loader
    """
    pin_memory = torch.cuda.is_available()

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        collate_fn=collate_fn
    )

    return dataloader