from sumie.utils.sequence_labeling_utils import get_labeling_scheme_from_name
from sumie.data.sample_and_split import load, save
from sumie.utils.experiment import ConfigFileExperiment
from sumie.utils.tokenization_utils import get_tokenizer_from_name
from sumie.data.annotation_utils import get_raw_wr_text_from_annotation, insert_linebreaks_for_annotation, insert_bullets_for_annotation
from sumie.models.definitions.transformer_sequence_labeler import TransformerSequenceLabeler
from transformers import logging, AutoModelForTokenClassification
from ast import literal_eval
import pandas as pd
import os
import glob
import torch

class InferenceWithTransformerSequenceLabeler(ConfigFileExperiment): 
    def __init__(self, config_name = 'predict_config', *args, **kwargs):
        super().__init__(config_name = config_name, *args, **kwargs)

    def setup(self):
        logging.set_verbosity_error() 
        
        predict_config = self.config[self.config_name]
        self.run_exp = predict_config['run_exp']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.labeling_scheme = get_labeling_scheme_from_name(predict_config['labeling_scheme'])
        self.data_schema_type = predict_config['data_schema_type']
        self.model = TransformerSequenceLabeler(predict_config['model_name'], num_labels=len(self.labeling_scheme.labels)).to(self.device)
        self.model.load(glob.glob(os.path.join(self.working_dir, 'train_checkpoints/best*'))[0])
        #self.model = AutoModelForTokenClassification.from_pretrained(predict_config['model_name'], num_labels=len(self.labeling_scheme.labels)).to(self.device)
        #self.model.load_state_dict(torch.load(glob.glob(os.path.join(self.working_dir, 'train_checkpoints/best*'))[0]))
        self.tokenizer = get_tokenizer_from_name(predict_config['tokenizer_name'])
        self.test_df = load(predict_config['test_data_path'])
        self.inference_df = pd.DataFrame(columns=['wh_id', 'wr_id', 'wr_text', 'predicted_annotation', 'ground_truth_annotation'])
    
    def run(self): 
        for i, row in self.test_df.iterrows():
             
            wh_id = row['wh_id']
            wr_id = row['wr_id']
            ground_truth_annotation = row['annotate']
            wr_text = None

            if self.data_schema_type == 'sequence_labeling_with_line_breaks':
                annotation_with_line_breaks = insert_linebreaks_for_annotation(ground_truth_annotation, literal_eval(row['wr_lines']))
                wr_text = get_raw_wr_text_from_annotation(annotation_with_line_breaks)

            elif self.data_schema_type == 'sequence_labeling_with_bullets':
                annotation_with_bullets = insert_bullets_for_annotation(ground_truth_annotation, literal_eval(row['wr_lines']), literal_eval(row['line_list_item_mask']))
                wr_text = get_raw_wr_text_from_annotation(annotation_with_bullets)

            elif self.data_schema_type == 'sequence_labeling_with_line_breaks_and_bullets':
                annotation_with_line_breaks = insert_linebreaks_for_annotation(ground_truth_annotation, literal_eval(row['wr_lines']))
                annotation_with_line_breaks_and_bullets = insert_bullets_for_annotation(annotation_with_line_breaks, literal_eval(row['wr_lines']), literal_eval(row['line_list_item_mask']))
                wr_text = get_raw_wr_text_from_annotation(annotation_with_line_breaks_and_bullets)

            elif self.data_schema_type == 'sequence_labeling':
                wr_text = get_raw_wr_text_from_annotation(ground_truth_annotation)

            else:
                raise ValueError(f'data_schema_type undefined for {self.__class__.__name__}!')
            
            tokenized_inputs = {k : v.to(self.device) for k, v in self.tokenizer([wr_text]).items()}
            predicted_labels = self.model(**tokenized_inputs)['logits'][0].argmax(axis=1).detach().cpu().numpy().tolist()
            predicted_annotation = self.labeling_scheme.annotate_text(wr_text, predicted_labels, self.tokenizer)
            wr_text = get_raw_wr_text_from_annotation(ground_truth_annotation)
            self.inference_df.loc[len(self.inference_df)] = [wh_id, wr_id, wr_text, predicted_annotation, ground_truth_annotation]

        save(self.inference_df, os.path.join(self.working_dir, "inference.csv"))

if __name__ == '__main__': 
    with InferenceWithTransformerSequenceLabeler() as exp: 
        exp.run()