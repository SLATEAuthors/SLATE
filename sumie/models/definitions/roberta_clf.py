import torch
import torch.nn as nn
from collections import OrderedDict

class RobertaWithClfHead(nn.Module):
    def __init__(self, roberta, config):
        super(RobertaWithClfHead, self).__init__()
        
        self.roberta = roberta
        self.classifier = nn.Sequential(OrderedDict([
            ('dropout_1', nn.Dropout(config['dropout_prob'])),
            ('dense', nn.Linear(in_features=config['linear_1_in_features']*config["no_of_pooling_layers"], out_features=config['linear_1_out_features'])),
            ('batchnorm', nn.BatchNorm1d(num_features=config['linear_1_out_features'])) if config["normalization"]=="batchnorm" else ('layernorm', nn.LayerNorm(config['linear_1_out_features'])),
            ('relu', nn.ReLU()) if config['activation']=='relu' else ('gelu', nn.GELU()),
            ('dropout_2', nn.Dropout(config['dropout_prob'])),
            ('out_proj', nn.Linear(in_features=config['linear_1_out_features'], out_features=config['num_labels']))
        ]))
        self.classifier.apply(self.init_weights)
    
    def forward(self, no_of_pooling_layers=1, **kwargs):
        inputs = kwargs
        
        if no_of_pooling_layers == 1:
            outputs = self.roberta(**inputs, output_hidden_states=True)
            hidden_states = outputs.last_hidden_state
            pooler = hidden_states[:, 0, :]
        else:
            outputs = self.roberta(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            pooling_layers = [l for l in range(-no_of_pooling_layers, 0)]
            pooler = torch.cat(tuple([hidden_states[i] for i in pooling_layers]), dim=-1)[:, 0, :]
        
        logits = self.classifier(pooler)
        return logits

    def init_weights(self,m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)