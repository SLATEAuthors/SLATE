import numpy as np
import torch
from sumie.datasets.sumie_base_dataset import SumieBaseDataset

class TextClassifierTrainDataset(SumieBaseDataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)

    def transform(self, data):
        """
        This function creates features and returns numpy array for Bag of words classifier model.
        """
        print('starting to transform data')
        data_transformed = []
        for sample in data:
            data_transformed.append([
                sample["sentences"],
                int(sample["labels"])
            ])

        #print(type(data_transformed))
        print('data transformed!')

        return data_transformed

    def process_sample(self, sample):
        return sample