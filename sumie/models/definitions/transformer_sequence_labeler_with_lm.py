from sumie.utils.sequence_labeling_utils import SequenceSplitHandler
from transformers import AutoModelForTokenClassification, AutoModelForMaskedLM
from typing import Dict
import torch.nn as nn
import torch

class TransformerSequenceLabelerWithLM(nn.Module): 
    def __init__(self, model_name: str, num_labels: int, max_input_seq_len=None): 
        super(TransformerSequenceLabelerWithLM, self).__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
        self.lm_model = AutoModelForMaskedLM.from_pretrained(model_name)
        encoder = getattr(self.model, self.get_encoder_name(model_name))
        setattr(self.lm_model, self.get_encoder_name(model_name), encoder)
        self.sequence_split_handler = SequenceSplitHandler()
        self.max_input_seq_len = max_input_seq_len if max_input_seq_len is not None else self.model.config.max_position_embeddings - 2
        self.num_labels = num_labels

    def split_long_inputs(self, input_dict: Dict): 
        return self.sequence_split_handler.split_input_dict(input_dict, self.max_input_seq_len)

    def merge_output_splits(self, output_dict: Dict): 
        return self.sequence_split_handler.merge_output_dict(output_dict, self.max_input_seq_len)

    def prepare_inputs(self, raw_input_dict: Dict):
        prepared_input_dict = self.split_long_inputs(raw_input_dict)
        return prepared_input_dict

    def process_outputs(self, raw_output_dict: Dict): 
        processed_output_dict = self.merge_output_splits(raw_output_dict)
        return processed_output_dict

    def forward(self, mode='sequence_labeling', **kwargs):
        if 'return_dict' in kwargs and not kwargs['return_dict']: 
                raise ValueError("return_dict must be True for TransformerSequenceLabeler forward!") 

        if mode == 'sequence_labeling':
            #return self.model(**kwargs)
            return self.process_outputs(self.model(**self.prepare_inputs(kwargs)))
        
        elif mode == 'lm': 
            return self.lm_model(**kwargs)
            

    def load(self, state_dict_path:str): 
        self.model.load_state_dict(torch.load(state_dict_path))

    @classmethod
    def get_encoder_name(self, model_name): 
        if 'roberta' in model_name: 
            return 'roberta'
        else: 
            KeyError(f'get_encoder unsupported for {model_name}!')