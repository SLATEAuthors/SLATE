from sumie.utils.experiment import ConfigFileExperiment
from sumie.utils.tokenization_utils import get_tokenizer_from_name, BaseTokenizer
from sumie.utils.sequence_labeling_utils import get_labeling_scheme_from_name
from sumie.models.definitions.roberta_clf import RobertaWithClfHead
from sumie.models.definitions.roberta_clf_and_lm import RobertaWithClfHeadandLMHead
from transformers import AutoModelForTokenClassification, RobertaModel
from transformers import logging
import onnxruntime
import torch
import os
import glob
import onnx
import numpy as np

class ExportToOnnx(ConfigFileExperiment): 
    def __init__(self, config_name = 'onnx_config', *args, **kwargs):
        super().__init__(config_name = config_name, *args, **kwargs)

    def create_model_instance(self, model_type: str):
        if 'sequence_labeling' in model_type:
            self.labeling_scheme = get_labeling_scheme_from_name(self.config['labeling_scheme'])
            return AutoModelForTokenClassification.from_pretrained(self.model_name, num_labels=len(self.labeling_scheme.labels))
        elif 'classification' in model_type:
            if self.model_name == 'roberta_with_clf_head':
                roberta_base = RobertaModel.from_pretrained(self.model_checkpoint)
                return RobertaWithClfHead(roberta_base, self.config)
            elif self.model_name == 'roberta_with_clf_head_and_lm_head':
                roberta_base = RobertaModel.from_pretrained(self.model_checkpoint)
                return RobertaWithClfHeadandLMHead(roberta_base, self.config)

        raise ValueError('Could not create model_instance. model_type provided is undefined!')

    def create_sample_input(self, input_format_type: str):
        wrs = ['Hello. This is a dummy input.', 'Schedule time to write another dummy input.'] 
        if input_format_type == 'default':
            return self.tokenizer(wrs)
    
    def create_input_for_parity_test(self, input_format_type: str):
        wrs = ['Hello. This is a dummy input.', 'Schedule time to write another dummy input.', 'Send email today!'] 
        if input_format_type == 'default':
            return self.tokenizer(wrs)
    
    def process_torch_model_output(self, output):
        if 'sequence_labeling' in self.model_type:
            return output['logits']
        else: 
            return output
        
    def get_dynamic_axes(self, model_type: str): 
        if 'sequence_labeling' in model_type:
            return {
                'input_ids' : {0 : 'batch_size', 1: 'max_seq_length'},
                'attention_mask' : {0: 'batch_size', 1: 'max_seq_length'},
                'output': {0 : 'batch_size', 1: 'max_seq_length'}
            }
        elif 'classification' in model_type:
            return {
                'input_ids' : {0 : 'batch_size', 1: 'max_seq_length'},
                'attention_mask' : {0: 'batch_size', 1: 'max_seq_length'},
                'output': {0 : 'batch_size'}
            }
        else:
            raise ValueError('Dynamic axes not defined for model_type provided!')

    def setup(self):
       logging.set_verbosity_error()
         
       self.model_param_path = glob.glob(os.path.join(self.working_dir, 'train_checkpoints/best*'))[0]
       self.onnx_path = os.path.join(self.working_dir, 'train_checkpoints/model.onnx')
       
       onnx_config = self.config[self.config_name]
       self.config = onnx_config
       self.run_exp = onnx_config['run_exp']
       self.model_type = onnx_config['model_type']
       self.model_name = onnx_config['model_name']
       self.model_checkpoint = onnx_config['model_checkpoint'] if 'model_checkpoint' in onnx_config else ''
       self.tokenizer = get_tokenizer_from_name(onnx_config['tokenizer_name'])
       self.input_format_type = onnx_config['input_format_type']
       self.model = self.create_model_instance(self.model_type)
       self.model.load_state_dict(torch.load(self.model_param_path))
       self.sample_input = self.create_sample_input(self.input_format_type)
       self.dynamic_axes = self.get_dynamic_axes(self.model_type)
       
    
    def run(self):
        input_names = list(self.sample_input.keys())
        inputs = tuple(self.sample_input[name] for name in input_names)

        #set model to inference mode
        self.model.eval()
        torch.onnx.export(self.model, 
            inputs,
            self.onnx_path, 
            export_params=True,
            input_names=input_names, 
            output_names=['output'],
            opset_version=11, 
            dynamic_axes=self.dynamic_axes)

        onnx_model = onnx.load(self.onnx_path)
        
        #test if the model structure is correct and has a valid schema
        onnx.checker.check_model(onnx_model)

        # compare ONNX Runtime and PyTorch results
        ort_session = onnxruntime.InferenceSession(self.onnx_path)
        model_inputs = self.create_input_for_parity_test(self.input_format_type)
        ort_inputs = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}
        ort_outs = ort_session.run(None, ort_inputs)
        torch_outs = None
        with torch.no_grad():
            torch_outs = self.process_torch_model_output(self.model(**model_inputs)).numpy()

        np.testing.assert_allclose(torch_outs, ort_outs[0], rtol=1e-03, atol=1e-05, \
            err_msg='Onnx model output did not match torch model output within specified tolerance!')

