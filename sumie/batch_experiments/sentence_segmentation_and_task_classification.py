from sumie.train.train_transformer_classifier import FinetuneTransformerClassifier
from sumie.predict.predict_transformer_classifier import PredictTransformerClassifier
from sumie.train.train_sequence_labeler import TrainTransformerSequenceLabeler
from sumie.predict.predict_sequence_labeler import InferenceWithTransformerSequenceLabeler
from sumie.post_process_experiments.sentence_segmentation_post_process_task_classification import SentenceSegmentationPostProcessForTaskClassification
from sumie.evaluate.evaluate_on_ink import Evaluate
from sumie.models.onnx_interfaces.onnx_exporter import ExportToOnnx
from sumie.utils.experiment import BatchConfigFileExperiment

class SentenceSegmentationAndTaskClassificationExperiment(BatchConfigFileExperiment): 

    @property
    def pipeline(self): 
        return [
            (TrainTransformerSequenceLabeler, 'sentence_seg_train_config'), 
            (InferenceWithTransformerSequenceLabeler, 'sentence_seg_predict_config'),
            (ExportToOnnx, 'sentence_seg_onnx_config'),
            (SentenceSegmentationPostProcessForTaskClassification, 'sentence_seg_post_process_config'),
            (FinetuneTransformerClassifier, 'train_config'), 
            (PredictTransformerClassifier, 'predict_config'), 
            (Evaluate, 'eval_config'), 
            (ExportToOnnx, 'onnx_config')
        ]

if __name__ == '__main__': 
    with SentenceSegmentationAndTaskClassificationExperiment() as exp: 
        exp.run()