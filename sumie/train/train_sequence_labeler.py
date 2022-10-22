from sumie.utils.sequence_labeling_utils import get_labeling_scheme_from_name, SequenceLabelingBatch
from sumie.utils.torch_utils import create_dataloader
from sumie.utils.tokenization_utils import get_tokenizer_from_name
from sumie.datasets.sequence_labeling_dataset import SequenceLabelingDataset
from sumie.utils.experiment import ConfigFileExperiment
from sumie.models.definitions.transformer_sequence_labeler import TransformerSequenceLabeler
from transformers import AutoModelForTokenClassification, AdamW
from transformers import logging
from functools import partial
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint
from torch.nn import CrossEntropyLoss
from collections import Counter
import torch
import os
import random
import numpy as np

class TrainTransformerSequenceLabeler(ConfigFileExperiment):
    def __init__(self, config_name = 'train_config', *args, **kwargs):
        super().__init__(config_name = config_name, *args, **kwargs)

    def setup(self):
        logging.set_verbosity_error()

        train_config = self.config[self.config_name]
        self.set_random_seed(train_config['random_seed'])
        self.run_exp = train_config['run_exp']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.labeling_scheme = get_labeling_scheme_from_name(train_config['labeling_scheme'])
        self.model = TransformerSequenceLabeler(train_config['model_name'], num_labels=len(self.labeling_scheme.labels)).to(self.device)
        #self.model = AutoModelForTokenClassification.from_pretrained(train_config['model_name'], num_labels=len(self.labeling_scheme.labels)).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=train_config['lr'])
        self.tokenizer = get_tokenizer_from_name(train_config['tokenizer_name'])
        self.train_set = SequenceLabelingDataset(self.labeling_scheme, self.tokenizer, data_schema_type=train_config['data_schema_type'], data_path=train_config['train_data_path'])
        self.val_set = SequenceLabelingDataset(self.labeling_scheme, self.tokenizer, data_schema_type=train_config['data_schema_type'], data_path=train_config['val_data_path'])
        self.train_loader = create_dataloader(self.train_set, train_config['batch_size'], shuffle=True, collate_fn=partial(SequenceLabelingBatch, labeling_scheme=self.labeling_scheme, tokenizer=self.tokenizer), num_workers=0)
        self.val_loader = create_dataloader(self.val_set, train_config['batch_size'], shuffle=False, collate_fn=partial(SequenceLabelingBatch, labeling_scheme=self.labeling_scheme, tokenizer=self.tokenizer), num_workers=0)
        self.num_epochs = train_config['num_epochs']
        self.log_interval = train_config['log_interval']
        self.checkpoint_interval = train_config['checkpoint_interval']

        self.loss_weights = train_config['loss_weights']
        if(isinstance(self.loss_weights, str)): 
            self.loss_weights = self.compute_loss_weights(self.loss_weights)
        self.loss = CrossEntropyLoss(weight=self.loss_weights)

        self.trainer = Engine(self.train_step)
        ProgressBar().attach(self.trainer, output_transform=lambda x: {'batch loss' : x})
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=self.log_interval), self.log_training_loss)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=self.log_interval), self.log_val_loss)
        checkpoint_handler = ModelCheckpoint(os.path.join(self.working_dir, 'train_checkpoints/'), 'best', atomic=False, score_function=partial(self.log_val_loss, neg=True, verbose=False), score_name='neg_val_loss')
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=self.checkpoint_interval), checkpoint_handler, {'model_checkpoint': self.model.model})

        self.evaluator = Engine(self.val_step)

    def set_random_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def compute_loss_weights(self, strategy):
        if strategy == 'class_weights':
            counter = Counter()
            for batch in self.train_loader: 
                for l in batch.label_ids_list[batch.attention_mask_list.bool()].flatten():
                    counter[l.item()] += 1
            weights = []
            max_count = counter.most_common(1)[0][1]
            for l in self.labeling_scheme.str_to_int_labels(self.labeling_scheme.labels):
                weights.append(max_count/counter[l])
            return torch.tensor(weights).to(self.device)
        else: 
            raise ValueError("The provided loss_weights strategy is undefined!")
            

    def train_step(self, engine, batch): 
        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(
            input_ids=batch.token_ids_list.to(self.device),
            attention_mask=batch.attention_mask_list.to(self.device),
            return_dict=True
        )['logits']

        loss = self.compute_loss(logits, batch)
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def compute_loss(self, logits, batch):
        labels = batch.label_ids_list.to(self.device)
        attention_mask = batch.attention_mask_list.to(self.device)
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.model.num_labels)
        #elements with labels = self.loss.ignore_index are ignored in loss/gradient computation. This is used for ignoring padding.
        active_labels = torch.where(active_loss, labels.view(-1), torch.tensor(self.loss.ignore_index).type_as(labels))
        loss = self.loss(active_logits, active_labels)
        return loss

    def val_step(self, engine, batch):
        self.model.eval()
        loss = None
        with torch.no_grad(): 
            logits = self.model(
                    input_ids=batch.token_ids_list.to(self.device),
                    attention_mask=batch.attention_mask_list.to(self.device),
                    return_dict=True
                )['logits']
            loss = self.compute_loss(logits, batch)
        return loss.item()

    def log_training_loss(self, engine, verbose=True, neg=False):
        self.evaluator.run(self.train_loader)
        if verbose:  
            print(f"Epoch[{engine.state.epoch}] Training Loss: {self.evaluator.state.output:.4f}")
        return self.evaluator.state.output if not neg else -self.evaluator.state.output
    
    def log_val_loss(self, engine, verbose=True, neg=False):
        self.evaluator.run(self.val_loader)
        if verbose: 
            print(f"Epoch[{engine.state.epoch}] Val Loss: {self.evaluator.state.output:.4f}")
        return self.evaluator.state.output if not neg else -self.evaluator.state.output
    

    def run(self):
        self.trainer.run(self.train_loader, max_epochs=self.num_epochs)

if __name__ == '__main__':
    with TrainTransformerSequenceLabeler() as exp: 
        exp.run()