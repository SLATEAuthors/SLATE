import numpy as np
import torch
from ast import literal_eval
from sumie.datasets.sumie_base_dataset import SumieBaseDataset
from sumie.data.annotation_utils import parse_annotations_for_text, create_wr_context_feature_for_sentence_with_labels, str_label_to_int

class TaskClassifierDataset(SumieBaseDataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)

    def transform(self, data):
        """
        This function creates features and labels for task classifier
        """
        transformed_data = []
        for i,row in data.iterrows():
            if self.data_schema_type == "task_classification":
                wh_id = row['wh_id']
                wr_id = row['wr_id']
                sentences, labels = parse_annotations_for_text(row['annotate'])
                for sentence, label in zip(sentences, str_label_to_int(labels)):
                    transformed_data.append([wh_id, wr_id, sentence, label])
            if self.data_schema_type == "task_classification_on_a0_a1":
                wh_id = row['wh_id']
                wr_id = row['wr_id']
                sentences, labels = parse_annotations_for_text(row['annotate'])
                for sentence, label in zip(sentences, str_label_to_int(labels, scheme='only_a0_and_a1')):
                    if label != -1:
                        transformed_data.append([wh_id, wr_id, sentence, label])
            elif self.data_schema_type == "task_classification_on_sentence_segmentation":
                wh_id = row['wh_id']
                wr_id = row['wr_id']
                sentences, labels = parse_annotations_for_text(row['predicted_annotation'])
                for sentence, label in zip(sentences, str_label_to_int(labels)):
                    transformed_data.append([wh_id, wr_id, sentence, label])
            elif self.data_schema_type == "task_classification_with_context":
                wh_id = row['wh_id']
                wr_id = row['wr_id']
                sentences, labels = parse_annotations_for_text(row['annotate'])
                sentences_with_context, labels = create_wr_context_feature_for_sentence_with_labels(row['annotate'])
                for sentence, sentence_with_context, label in zip(sentences, sentences_with_context, str_label_to_int(labels)):
                    transformed_data.append([wh_id, wr_id, sentence_with_context, label, sentence])
            elif self.data_schema_type == "task_classification_on_lines":
                wh_id = row['wh_id']
                wr_id = row['wr_id']
                sentences = literal_eval(row['wr_lines'])
                for sentence in sentences:
                    transformed_data.append([wh_id, wr_id, sentence]) 
        return transformed_data

    def process_sample(self, sample):
        return sample