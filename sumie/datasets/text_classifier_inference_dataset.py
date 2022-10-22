import numpy as np
from sumie.datasets.sumie_base_dataset import SumieBaseDataset

class TextClassifierInferenceDataset(SumieBaseDataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)

    def transform(self, data):
        """
        This function creates features and returns numpy array.
        """
        print('starting to transform data')
        data_transformed = np.array([], dtype=object)
        for sample in data:
            data_transformed = np.concatenate((
                data_transformed,
                sample["sentences"]
            ), axis=None)
        
        print('data transformed!')

        return data_transformed

    def process_sample(self, sample):
        return sample