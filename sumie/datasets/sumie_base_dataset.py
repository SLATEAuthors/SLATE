from sumie.data.sample_and_split import load
import abc
import csv
import torch.utils.data

class SumieBaseDataset(torch.utils.data.Dataset):
    """
    Base dataset class that provides functionality to load data and format the data 
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data_path = kwargs["data_path"]
        self.data_schema_type = kwargs["data_schema_type"]
        data = self.load_data()

        self.data = self.transform(data)

    def load_data(self):
        data = []
        if self.data_schema_type == "simple_classifier_inference":
            field_names = ["wh_id", "wr_id", "sentences"]

            with open(self.data_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for i,row in enumerate(reader):
                    sample = {}
                    for field_name in field_names:
                        sample[field_name] = row[field_name]
                    data.append(sample)
        elif self.data_schema_type == "simple_classifier_training":
            field_names = ["wh_id", "wr_id", "sentences", "labels"]

            with open(self.data_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for i,row in enumerate(reader):
                    sample = {}
                    for field_name in field_names:
                        sample[field_name] = row[field_name]
                    data.append(sample)
        elif self.data_schema_type == "sequence_labeling": 
            df = load(self.data_path)
            data = df['annotate'].tolist()
        elif self.data_schema_type == "sequence_labeling_with_line_breaks":
            df = load(self.data_path)
            data = df[['annotate', 'wr_lines']]
        elif self.data_schema_type in ["sequence_labeling_with_bullets", "sequence_labeling_with_line_breaks_and_bullets"]:
            df = load(self.data_path)
            data = df[['annotate', 'wr_lines', 'line_list_item_mask']]
        elif self.data_schema_type in ("task_classification","task_classification_on_a0_a1"):
            df = load(self.data_path)
            data = df[['wh_id', 'wr_id', 'annotate']]
        elif self.data_schema_type == "task_classification_on_sentence_segmentation":
            df = load(self.data_path)
            data = df[['wh_id', 'wr_id', 'predicted_annotation']]
        elif self.data_schema_type == "task_classification_with_context":
            df = load(self.data_path)
            data = df[['wh_id', 'wr_id', 'annotate']]
        elif self.data_schema_type == "task_classification_on_lines":
            df = load(self.data_path)
            data = df[['wh_id', 'wr_id', 'wr_lines']]
        return data


    @abc.abstractmethod
    def transform(self, data):
        """
        Reformat raw data. 
        To be defined in child classes
        """
        pass

    @abc.abstractmethod
    def process_sample(self, sample):
        """
        Process sample and make it ready to be passed into the model. 
        Is used by __getitem__ 
        To be defined in child classes
        """
        pass

    def __getitem__(self, index):
        """Return processed sample to be consumed by the DataLoader""" 
        sample = self.data[index]
        sample_processed = self.process_sample(sample)
        return sample_processed

    def __len__(self):
        """Return the length of the dataset"""
        return len(self.data)