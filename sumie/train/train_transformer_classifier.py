import os
import sys
import re
import glob
import numpy as np
import torch
import torch.nn as nn
import argparse
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from transformers import RobertaModel
from sumie.utils.tokenization_utils import get_tokenizer_from_name
from sumie.utils.torch_utils import create_dataloader
from sumie.utils.experiment import ConfigFileExperiment
from sumie.datasets.text_classifier_train_dataset import TextClassifierTrainDataset
from sumie.datasets.task_classifier_dataset import TaskClassifierDataset
from sumie.models.definitions.roberta_clf import RobertaWithClfHead
from sumie.models.definitions.roberta_clf_and_lm import RobertaWithClfHeadandLMHead
from sumie.utils.io import save_pickle, save_list
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import ProgressBar, CosineAnnealingScheduler, PiecewiseLinear, create_lr_scheduler_with_warmup
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.contrib.handlers.tensorboard_logger import *

class FinetuneTransformerClassifier(ConfigFileExperiment):
    """
    This class implements training of transformer for sequence classification.
    """
    def __init__(self, config_name = 'train_config', *args, **kwargs):
        super().__init__(config_name = config_name, *args, **kwargs)

    def setup(self):
        self.config = self.config[self.config_name]
        self.set_random_seed(self.config['random_seed'])
        self.model = None
        self.tokenizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset_train = None
        self.dataloader_train = None
        self.dataset_val = None
        self.dataloader_val = None
        self.trainer = Engine(self.train_step)
        self.evaluator = Engine(self.validation_step)
        self.optimizer = None
        self.clf_loss_func = None
        self.lm_loss_func = None
        self.run_exp = self.config['run_exp']

    def set_random_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def load_data(self):
        dataset_train_params = dict(data_path=os.path.join(self.experiment_dir, "train", self.config["train_data_file_name"]), data_schema_type=self.config["train_data_schema_type"], **self.config)
        self.dataset_train = TaskClassifierDataset(**dataset_train_params)
        self.dataloader_train = torch.utils.data.DataLoader(self.dataset_train, self.config["batch_size"], shuffle=True, num_workers=1)
        print("Train dataset loaded!")
        print(f'Total samples: {len(self.dataset_train)}')

        dataset_val_params = dict(data_path=os.path.join(self.experiment_dir, "train", self.config["val_data_file_name"]), data_schema_type=self.config["train_data_schema_type"], **self.config)
        self.dataset_val = TaskClassifierDataset(**dataset_val_params)
        self.dataloader_val = torch.utils.data.DataLoader(self.dataset_val, self.config["batch_size"], shuffle=True, num_workers=1)
        print("Val dataset loaded!")
        print(f'Total samples: {len(self.dataset_val)}')

    def load_model(self):
        roberta_base = RobertaModel.from_pretrained(self.config['model_checkpoint'])
        if self.config["model_name"] == "roberta_with_clf_head":
            self.model = RobertaWithClfHead(roberta_base, self.config)
        elif self.config["model_name"] == "roberta_with_clf_head_and_lm_head":
            self.model = RobertaWithClfHeadandLMHead(roberta_base, self.config)
        self.model = self.model.to(self.device)
        print(f'model loaded onto {self.device}!')

    def load_tokenizer(self, use_fast=True):
        self.tokenizer = get_tokenizer_from_name(self.config['tokenizer_name'])
    
    def compute_loss_weights(self):
        labels = []
        for batch in self.dataloader_train:
            labels.extend(list(batch[3]))
        # since labels are tensors we extract its value
        for i in range(len(labels)):
            labels[i] = labels[i].item()
        unique_labels = set(labels)
        tot_labels = len(labels)
        class_weights = [0]*len(unique_labels)
        # cal count per label
        for val in labels:
            class_weights[val] += 1
        # compute weight by inversing count
        for i in range(len(class_weights)):
            class_weights[i] = tot_labels/class_weights[i]
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        print(f"Compute loss weights->{class_weights}")
        return class_weights

    def train_step(self, engine, batch):
        self.model.train()
        self.optimizer.zero_grad()
        inputs = self.tokenizer(list(batch[2]), self.device)
        if self.config['loss_type'] == 'clf_loss':
            labels = torch.tensor(list(batch[3])).to(self.device)
            logits = self.model(**inputs, no_of_pooling_layers=self.config["no_of_pooling_layers"])
            loss = self.clf_loss_func(logits, labels)
        elif self.config['loss_type'] == 'clf_and_lm_loss':
            clf_labels = torch.tensor(list(batch[3])).to(self.device)
            lm_labels = inputs["input_ids"]
            clf_logits, lm_logits = self.model(**inputs, no_of_pooling_layers=self.config["no_of_pooling_layers"])
            clf_loss = self.clf_loss_func(clf_logits, clf_labels)
            lm_loss = self.lm_loss_func(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            loss = self.config['clf_loss_coef']*clf_loss + self.config['lm_loss_coef']*lm_loss

        loss.backward()
        if self.config["clip_grad_norm"]:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_norm"])
        self.optimizer.step()
        return loss.item()

    def validation_step(self, engine, batch):
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(list(batch[2]), self.device)
            if self.config['loss_type'] == 'clf_loss':
                labels = torch.tensor(list(batch[3])).to(self.device)
                logits = self.model(**inputs, no_of_pooling_layers=self.config["no_of_pooling_layers"])
                loss = self.clf_loss_func(logits, labels)
            elif self.config['loss_type'] == 'clf_and_lm_loss':
                clf_labels = torch.tensor(list(batch[3])).to(self.device)
                lm_labels = inputs["input_ids"]
                clf_logits, lm_logits = self.model(**inputs, no_of_pooling_layers=self.config["no_of_pooling_layers"])
                clf_loss = self.clf_loss_func(clf_logits, clf_labels)
                #lm_loss = self.lm_loss_func(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
                loss = clf_loss

            return loss.item()

    def freeze_layers(self):
        for param in self.model.roberta.parameters():
            param.requires_grad = False
        full_parameters = sum(p.numel() for p in self.model.parameters())
        trained_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Training {trained_parameters:3e} parameters out of {full_parameters:3e},i.e. {100 * trained_parameters/full_parameters:.2f}%")

    def freeze_inside_layers(self):
        for name, param in self.model.named_parameters():
            if 'embeddings' not in name and 'classifier' not in name:
                param.detach_()
                param.requires_grad = False
            else:
                param.requires_grad = True

        full_parameters = sum(p.numel() for p in self.model.parameters())
        trained_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
        print(f"We will start by training {trained_parameters:3e} parameters out of {full_parameters:3e},"
            f" i.e. {100 * trained_parameters/full_parameters:.2f}%")

    def gradual_unfreezing(self, engine):
        unfreezing_interval = int(len(self.dataloader_train) * self.config['epochs'] / (self.config['num_layers'] + 1))

        if engine.state.iteration % unfreezing_interval == 0:
            # layer to unfreeze 
            unfreezing_index = self.config['num_layers'] - (engine.state.iteration // unfreezing_interval)

            # unfreeze layer!
            unfreezed = []
            for name, param in self.model.named_parameters():
                if re.match(r"roberta\.[^\.]*\.layer\." + str(unfreezing_index) + r"\.", name):
                    unfreezed.append(name)
                    param.require_grad = True
            print(f"Unfreezing block {unfreezing_index} with {unfreezed}")

    def update_layer_learning_rates(self, engine):
        for param_group in self.optimizer.param_groups:
            layer_index = int(param_group["name"])
            param_group["lr"] = param_group["lr"] / (self.config['decreasing_factor'] ** layer_index)

    def create_optimizer(self):
        if self.config['discriminative_learning']:
            parameter_groups = []
            for i in range(self.config['num_layers']):
                name_pattern = r"roberta\.[^\.]*\.layer\." + str(i) + r"\."
                group = {'name': str(self.config['num_layers'] - i),
                        'params': [p for n, p in self.model.named_parameters() if re.match(name_pattern, n)]}
                parameter_groups.append(group)

            # Add the rest of the parameters (embeddings and classification layer) in a group labeled '0'
            name_pattern = r"roberta\.[^\.]*\.layer\.\d*\."
            group = {'name': '0',
                    'params': [p for n, p in self.model.named_parameters() if not re.match(name_pattern, n)]}
            parameter_groups.append(group)

            # Sanity check that we still have the same number of parameters
            assert sum(p.numel() for g in parameter_groups for p in g['params'])\
                == sum(p.numel() for p in self.model.parameters())
            
            if self.config["optimizer"] == "Adam":
                optimizer = torch.optim.Adam(parameter_groups, lr=self.config["lr"], betas=(self.config["beta1"], self.config["beta2"]), eps=self.config["eps"], weight_decay=self.config["weight_decay"])
        else:
            if self.config["optimizer"] == "Adam":
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"], betas=(self.config["beta1"], self.config["beta2"]), eps=self.config["eps"], weight_decay=self.config["weight_decay"])
        
        return optimizer

    def train(self):
        # define loss function
        if self.config["loss_weights"]:
            class_weights = self.compute_loss_weights()
            self.clf_loss_func = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.clf_loss_func = nn.CrossEntropyLoss()
        if self.config["loss_type"] == "clf_and_lm_loss":
            self.lm_loss_func = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        # define optimizer
        self.optimizer = self.create_optimizer()

        if self.config["piecewise_linear_scheduler"]:
            # Learning rate schedule: linearly warm-up to lr and then to zero
            scheduler = PiecewiseLinear(self.optimizer, 'lr', literal_eval(self.config["piecewise_linear_scheduler"]))
            self.trainer.add_event_handler(Events.EPOCH_STARTED, scheduler)

        if self.config["freeze_base_layers"]:
            self.trainer.add_event_handler(Events.STARTED, self.freeze_layers)

        if self.config["freeze_inside_layers"]:
            self.trainer.add_event_handler(Events.STARTED, self.freeze_inside_layers)

        @self.trainer.on(Events.STARTED)
        def start_message():
            print("Training started!")

        if self.config['discriminative_learning']:
            self.trainer.add_event_handler(Events.ITERATION_STARTED, self.update_layer_learning_rates)

        RunningAverage(output_transform=lambda x: x).attach(self.trainer, "loss")
        ProgressBar(persist=True).attach(self.trainer, metric_names=['loss'])

        if self.config['gradual_unfreezing']:
            self.trainer.add_event_handler(Events.ITERATION_COMPLETED, self.gradual_unfreezing)

        @self.trainer.on(Events.COMPLETED)
        def end_message():
            print("Training completed!")

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def run_validation():
            self.evaluator.run(self.dataloader_val, max_epochs=1)

        RunningAverage(output_transform=lambda x: x).attach(self.evaluator, "loss")

        @self.evaluator.on(Events.COMPLETED)
        def log_validation_results():
            metrics = self.evaluator.state.metrics
            print("Validation Results - Epoch: {}  Avg loss: {:.2f} "
                .format(self.trainer.state.epoch, metrics["loss"]))

        early_stopping_handler = EarlyStopping(patience=self.config['patience'], score_function=lambda engine: -engine.state.metrics["loss"], trainer=self.trainer)
        self.evaluator.add_event_handler(Events.COMPLETED, early_stopping_handler)
        #gst = lambda *_: self.trainer.state.epoch
        checkpoint_handler = ModelCheckpoint(os.path.join(self.working_dir, "train_checkpoints"), 'best', score_function=lambda engine: -engine.state.metrics["loss"], score_name='neg_val_loss', n_saved=self.config['no_models_saved'], atomic=False, require_empty=False)
        self.evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'model_checkpoint': self.model})

        # Create a logger
        tb_logger = TensorboardLogger(log_dir=os.path.join(self.working_dir, "tensorboard_logging"))

        if self.config["viz_loss"]:
            # Attach the logger to the trainer to log training loss at each iteration
            tb_logger.attach_output_handler(
                self.trainer,
                event_name=Events.EPOCH_COMPLETED,
                tag="training",
                output_transform=lambda loss: {"loss": loss}
            )

            # Attach logger to evaluator for validation loss logging
            tb_logger.attach_output_handler(
                self.evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="validation",
                metric_names=["loss"],
                global_step_transform=global_step_from_engine(self.trainer)
            )

        if self.config["viz_lr"]:
            # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
            tb_logger.attach_opt_params_handler(
                self.trainer,
                event_name=Events.EPOCH_COMPLETED,
                optimizer=self.optimizer,
                param_name='lr'  # optional
            )

        if self.config["viz_weights_norm"]: 
            # Attach the logger to the trainer to log model's weights norm after each iteration
            tb_logger.attach(
                self.trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsScalarHandler(self.model)
            )

        if self.config["viz_weights_hist"]:
            # Attach the logger to the trainer to log model's weights as a histogram after each epoch
            tb_logger.attach(
                self.trainer,
                event_name=Events.EPOCH_COMPLETED,
                log_handler=WeightsHistHandler(self.model)
            )

        if self.config["viz_grads_norm"]:
            # Attach the logger to the trainer to log model's gradients norm after each iteration
            tb_logger.attach(
                self.trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsScalarHandler(self.model)
            )

        if self.config["viz_grads_hist"]:
            # Attach the logger to the trainer to log model's gradients as a histogram after each epoch
            tb_logger.attach(
                self.trainer,
                event_name=Events.EPOCH_COMPLETED,
                log_handler=GradsHistHandler(self.model)
            )

        self.trainer.run(self.dataloader_train, max_epochs=self.config["epochs"])

        # We need to close the logger with we are done
        tb_logger.close()
    
    def run(self):
        self.load_data()
        self.load_model()
        self.load_tokenizer()
        self.train()
    
def main():
    with FinetuneTransformerClassifier() as exp:
        exp.run()
        
if __name__ == "__main__":
    main()


    


