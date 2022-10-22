import os
import glob
import numpy as np
import torch
from tqdm import tqdm
import argparse
import pandas as pd
from ast import literal_eval
from sumie.utils.torch_utils import create_dataloader
from sumie.utils.experiment import ConfigFileExperiment
from sumie.datasets.text_classifier_inference_dataset import TextClassifierInferenceDataset
from sumie.datasets.task_classifier_dataset import TaskClassifierDataset
from sumie.models.definitions.roberta_clf import RobertaWithClfHead
from sumie.models.definitions.roberta_clf_and_lm import RobertaWithClfHeadandLMHead
from sumie.utils.tokenization_utils import get_tokenizer_from_name
from transformers import RobertaModel
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import ProgressBar

class PredictTransformerClassifier(ConfigFileExperiment):
    """
    This class implements inference of task classification model.
    """
    def __init__(self, config_name = 'predict_config', *args, **kwargs):
        super().__init__(config_name = config_name, *args, **kwargs)

    def setup(self):
        self.dataset_test = None
        self.dataloader_test = None
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.wh_id, self.wr_id, self.sentences, self.predictions = [], [], [], []
        self.config = self.config[self.config_name]
        self.test_data_path = None
        # used as suffix for prediction file
        self.test_data_file_name = None
        self.run_exp = self.config['run_exp']
        self.data_schema_type = self.config['data_schema_type']

    def load_model(self):
        roberta_base = RobertaModel.from_pretrained(self.config['model_checkpoint'])
        if self.config["model_name"] == "roberta_with_clf_head":
            self.model = RobertaWithClfHead(roberta_base, self.config)
        elif self.config["model_name"] == "roberta_with_clf_head_and_lm_head":
            self.model = RobertaWithClfHeadandLMHead(roberta_base, self.config)
        checkpoint = torch.load(glob.glob(os.path.join(self.working_dir, 'train_checkpoints/best*'))[0])
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        print(f"Model loaded onto {self.device}!")

    def load_tokenizer(self):
        self.tokenizer = get_tokenizer_from_name(self.config['tokenizer_name'])

    def load_data(self):
        dataset_test_params = dict(data_path=self.test_data_path, **self.config)
        # update data schema based on lines, groundtruth, sentences
        dataset_test_params['data_schema_type'] = self.data_schema_type
        self.dataset_test = TaskClassifierDataset(**dataset_test_params)
        self.dataloader_test = torch.utils.data.DataLoader(self.dataset_test, self.config["batch_size"], shuffle=False, num_workers=1)
        print("Test dataset")
        print(f'Total samples: {len(self.dataset_test)}')

    def post_process_and_save_results(self):
        print(f"Postprocessing data...")
        test_df = pd.DataFrame({
            'wh_id': self.wh_id,
            'wr_id': self.wr_id,
            'sentences': self.sentences,
            'predictions': self.predictions
        })
        
        test_df["predicted_annotation"] = np.where(test_df["predictions"]==0, "<s>"+test_df["sentences"]+"<a0>", "<s>"+test_df["sentences"]+"<a1>")
        test_df.drop('predictions', 1, inplace=True)
        test_df = (test_df.groupby(['wh_id','wr_id'], sort=False)
          .agg({'predicted_annotation': lambda x: "".join(str(text) for text in x)})
          .reset_index())
        
        gold_test_df = pd.read_csv(self.config["gold_test_data_path"])

        eval_df = pd.merge(test_df, gold_test_df, on=["wh_id", "wr_id"], how="inner")
        eval_df["wr_text"] = eval_df['wr_lines'].apply(lambda x: " ".join(literal_eval(x)))
        eval_df.rename(columns = {"annotate": "ground_truth_annotation"}, inplace = True)
        columns_to_keep = ["wh_id", "wr_id", "wr_text", "predicted_annotation", "ground_truth_annotation"]
        eval_df = eval_df[columns_to_keep]

        eval_df.to_csv(os.path.join(self.working_dir, "inference_"+self.test_data_file_name+".csv"), index=False)
    
    def inference(self, engine, batch):
        # append wh_id, wr_id, sentences
        self.wh_id.extend(list(batch[0]))
        self.wr_id.extend(batch[1].tolist())
        if self.config['data_schema_type'] in ("task_classification", "task_classification_on_lines"):
            self.sentences.extend(list(batch[2]))
        elif self.config['data_schema_type'] == "task_classification_with_context":
            self.sentences.extend(list(batch[4]))

        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(list(batch[2]), self.device)
            if self.config['loss_type'] == 'clf_loss':
                logits = self.model(**inputs, no_of_pooling_layers=self.config["no_of_pooling_layers"])
            elif self.config['loss_type'] == 'clf_and_lm_loss':
                clf_logits, lm_logits = self.model(**inputs, no_of_pooling_layers=self.config["no_of_pooling_layers"])
                logits = clf_logits
            prob = torch.nn.Softmax(dim=1)(logits)[:,1].cpu().tolist()
            return prob

    def run(self):
        self.load_model()
        self.load_tokenizer()
        test_data_file_names = self.config['test_data_file_names'].split(',')
        for test_data_file_name in test_data_file_names:
            self.test_data_file_name = test_data_file_name
            if self.test_data_file_name == "groundtruth":
                self.test_data_path = self.config["gold_test_data_path"]
                self.data_schema_type = "task_classification"
            elif self.test_data_file_name == "lines":
                if self.config['data_schema_type'] == "task_classification":
                    # update schema for lines
                    self.data_schema_type = "task_classification_on_lines"
                    self.test_data_path = self.config["gold_test_data_path"]
            elif self.test_data_file_name == "sentences":
                if self.config['data_schema_type'] == "task_classification":
                    # update schema for lines
                    self.data_schema_type = "task_classification_on_sentence_segmentation"
                    self.test_data_path = os.path.join(self.working_dir, f'predicted_{self.test_data_file_name}.csv')

            print(f'Inference for {test_data_file_name}...')
            self.wh_id, self.wr_id, self.sentences, self.predictions = [], [], [], []
            self.load_data()
            self.evaluator = Engine(self.inference)

            @self.evaluator.on(Events.STARTED)
            def start_message():
                print("Inference started!")

            ProgressBar(persist=True).attach(self.evaluator)

            @self.evaluator.on(Events.ITERATION_COMPLETED)
            def update_predictions():
                prob = self.evaluator.state.output
                pred = [1 if x>0.5 else 0 for x in prob]
                self.predictions.extend(pred)

            @self.evaluator.on(Events.COMPLETED)
            def end_message():
                print("Training completed!")

            self.evaluator.run(self.dataloader_test, max_epochs=1)

            self.post_process_and_save_results()


def main():
    with PredictTransformerClassifier() as exp:
        exp.run()
        
if __name__ == "__main__":
    main()