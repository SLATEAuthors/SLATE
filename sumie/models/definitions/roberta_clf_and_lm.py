import torch
import torch.nn as nn
from collections import OrderedDict

class RobertaWithClfHeadandLMHead(nn.Module):
    def __init__(self, roberta, config):
        super(RobertaWithClfHeadandLMHead, self).__init__()
        
        self.roberta = roberta
        self.classifier = nn.Sequential(OrderedDict([
            ('dropout_1', nn.Dropout(config['dropout_prob'])),
            ('dense', nn.Linear(in_features=config['linear_1_in_features']*config["no_of_pooling_layers"], out_features=config['linear_1_out_features'])),
            ('relu', nn.ReLU()) if config['activation']=='relu' else ('gelu', nn.GELU()),
            ('batchnorm', nn.BatchNorm1d(num_features=config['linear_1_out_features'])) if config["normalization"]=="batchnorm" else ('layernorm', nn.LayerNorm(config['linear_1_out_features'])),
            ('dropout_2', nn.Dropout(config['dropout_prob'])),
            ('out_proj', nn.Linear(in_features=config['linear_1_out_features'], out_features=config['num_labels']))
        ]))
        self.classifier.apply(self.init_weights)

        self.lm_head = nn.Sequential(OrderedDict([
            ('dense', nn.Linear(in_features=config['linear_1_in_features'], out_features=config['linear_1_out_features'])),
            ('relu', nn.ReLU()) if config['activation']=='relu' else ('gelu', nn.GELU()),
            ('batchnorm', nn.BatchNorm1d(num_features=config['linear_1_out_features'])) if config["normalization"]=="batchnorm" else ('layernorm', nn.LayerNorm(config['linear_1_out_features'])),
            ('decoder', nn.Linear(in_features=config['linear_1_out_features'], out_features=config['num_embeddings'], bias=False))
        ]))

        self.lm_head.apply(self.init_weights)
        self.init_decoder_weights()
    
    def forward(self, input_ids=None, attention_mask=None, no_of_pooling_layers=1):
        inputs = self.prepare_inputs(input_ids, attention_mask)
        
        if no_of_pooling_layers == 1:
            outputs = self.roberta(**inputs)
            last_hidden_state = outputs.last_hidden_state
            pooler = last_hidden_state[:, 0, :]
        else:
            outputs = self.roberta(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.last_hidden_state
            hidden_states = outputs.hidden_states
            pooling_layers = [l for l in range(-no_of_pooling_layers, 0)]
            pooler = torch.cat(tuple([hidden_states[i] for i in pooling_layers]), dim=-1)[:, 0, :]
        
        clf_logits = self.classifier(pooler)
        lm_logits = self.lm_head(last_hidden_state)
        return clf_logits, lm_logits

    def prepare_inputs(self, input_ids, attention_mask):
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def init_decoder_weights(self):
        self.lm_head.decoder.weight = self.roberta.embeddings.word_embeddings.weight