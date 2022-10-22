from sumie.train.train_sequence_labeler_with_lm_loss import TrainTransformerSequenceLabelerWithLMLoss
from sumie.predict.predict_sequence_labeler import InferenceWithTransformerSequenceLabeler
from sumie.evaluate.evaluate_on_ink import Evaluate
from sumie.models.onnx_interfaces.onnx_exporter import ExportToOnnx
from sumie.utils.experiment import BatchConfigFileExperiment

class SequenceLabelingWithLMLossBatchExperiment(BatchConfigFileExperiment): 

    @property
    def pipeline(self): 
        return [
            (TrainTransformerSequenceLabelerWithLMLoss, 'train_config'), 
            (InferenceWithTransformerSequenceLabeler, 'predict_config'), 
            (Evaluate, 'eval_config'), 
            (ExportToOnnx, 'onnx_config')
        ]

if __name__ == '__main__': 
    with SequenceLabelingWithLMLossBatchExperiment() as exp: 
        exp.run()