import numpy as np
from sumie.datasets.sumie_base_dataset import SumieBaseDataset
from sumie.utils.tokenization_utils import BaseTokenizer
from sumie.utils.sequence_labeling_utils import BaseSequenceLabelingScheme, get_labeling_scheme_from_name
from sumie.data.annotation_utils import insert_linebreaks_for_annotation, insert_bullets_for_annotation
from typing import Union
from ast import literal_eval

class SequenceLabelingDataset(SumieBaseDataset):
    def __init__(self, labeling_scheme: Union[str, BaseSequenceLabelingScheme], tokenizer: BaseTokenizer, *args,**kwargs):
        super().__init__(*args, **kwargs)
        
        self.tokenizer = tokenizer
        self.labeling_scheme = labeling_scheme

        if(isinstance(labeling_scheme, str)): 
            self.labeling_scheme = get_labeling_scheme_from_name(labeling_scheme) 

    def transform(self, data):
        if self.data_schema_type == 'sequence_labeling':
            return data
        elif self.data_schema_type == 'sequence_labeling_with_line_breaks':
            transformed_data = []
            for i,row in data.iterrows():
                annotated_wr = row['annotate']
                wr_lines = literal_eval(row['wr_lines'])
                transformed_data.append(insert_linebreaks_for_annotation(annotated_wr, wr_lines))
            return transformed_data

        elif self.data_schema_type == 'sequence_labeling_with_bullets':
            transformed_data = []
            for i,row in data.iterrows():
                annotated_wr = row['annotate']
                wr_lines = literal_eval(row['wr_lines'])
                bullet_mask = literal_eval(row['line_list_item_mask'])
                transformed_data.append(insert_bullets_for_annotation(annotated_wr, 
                    wr_lines,
                    bullet_mask))
            return transformed_data

        elif self.data_schema_type == 'sequence_labeling_with_line_breaks_and_bullets':
            transformed_data = []
            for i,row in data.iterrows():
                annotated_wr = row['annotate']
                wr_lines = literal_eval(row['wr_lines'])
                bullet_mask = literal_eval(row['line_list_item_mask'])
                transformed_data.append(insert_bullets_for_annotation(insert_linebreaks_for_annotation(annotated_wr, wr_lines), 
                    wr_lines,
                    bullet_mask))
            return transformed_data

        else:
            raise KeyError(f'Transform type for {self.__class__.__name__} is undefined!')

    def process_sample(self, sample):
        return sample