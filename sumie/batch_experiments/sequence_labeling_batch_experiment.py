from sumie.train.train_sequence_labeler import TrainTransformerSequenceLabeler
from sumie.predict.predict_sequence_labeler import InferenceWithTransformerSequenceLabeler
from sumie.evaluate.evaluate_on_ink import Evaluate
from sumie.models.onnx_interfaces.onnx_exporter import ExportToOnnx
from sumie.utils.experiment import BatchConfigFileExperiment

class SequenceLabelingBatchExperiment(BatchConfigFileExperiment): 

    @property
    def pipeline(self): 
        return [
            (TrainTransformerSequenceLabeler, 'train_config'), 
            (InferenceWithTransformerSequenceLabeler, 'predict_config'), 
            (Evaluate, 'eval_config'), 
            (ExportToOnnx, 'onnx_config')
        ]

if __name__ == '__main__': 
    with SequenceLabelingBatchExperiment() as exp: 
        exp.run()