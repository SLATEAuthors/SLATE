import os
import shutil
from sumie.utils.experiment import ConfigFileExperiment

class SentenceSegmentationPostProcessForTaskClassification(ConfigFileExperiment):
    def __init__(self, config_name = 'sentence_seg_post_process_config', *args, **kwargs):
        super().__init__(config_name = config_name, *args, **kwargs)

    def setup(self):
        self.config = self.config[self.config_name]
        self.train_checkpoint_dir = os.path.join(self.working_dir,"train_checkpoints")
        self.sentence_seg_dir = os.path.join(self.train_checkpoint_dir, "sentence_segmentation")
        self.inference_file_name = os.path.join(self.working_dir,f"predicted_{self.config['sentence_seg_suffix']}.csv")

    def run(self):
        
        # rename inference.csv
        os.rename(os.path.join(self.working_dir,"inference.csv"), self.inference_file_name)

        
        # move model checkpoint and onnx file
        file_names = os.listdir(self.train_checkpoint_dir)
        os.mkdir(self.sentence_seg_dir)
        for file_name in file_names:
            shutil.move(os.path.join(self.train_checkpoint_dir, file_name), self.sentence_seg_dir)

        print('Renamed inference file to {self.inference_file_name} and moved model checkpoints to {self.sentence_seg_dir}')



