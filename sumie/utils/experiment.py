import argparse
import json
import os
import shutil
import glob
import subprocess
from abc import abstractmethod
from typing import Dict

from sumie.utils.io import save_json, load_json


class ExperimentLoadError(Exception):
    """
    Error raised if there were error loading an experiment
    """
    pass

class Experiment():
    """
    Base class for all experiments. Parses command line arguments.
    """
    CONFIG_FILENAME = 'config.json'

    def __init__(self, experiment_dir=None, experiment_name=None, run_name=None):

        if not (experiment_dir is None) and (experiment_name is None) and (run is None):
            run_dir = os.path.join(experiment_dir, experiment_name, "run_" + str(run_name))
            # load existing experiment
            config_filename = os.path.join(run_dir, Experiment.CONFIG_FILENAME)

            if not os.path.exists(config_filename):
                raise ExperimentLoadError(f'Cannot load experiment from {run_dir}')

            config = load_json(config_filename)
            
        else:
            # create a new experiment
            config = self._parse_args()
            if 'config' in config:
                config = load_json(config['config'])
                
            experiment_dir = config['experiment_dir']
            experiment_name = config['experiment_name']
            run_name = config['run']

            run_dir = os.path.join(experiment_dir, experiment_name, "run_" + str(run_name))
            # create dir if not exists
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)

        self.experiment_dir = experiment_dir
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run_dir = run_dir
        self.config = config

    def _parse_args(self):
        """Parse the command line arguments.
        
        Returns:
            dictionary -- A dictionary that contains command line arguments
        """
        parser = argparse.ArgumentParser(description=self.experiment_name)
        parser.add_argument('--experiment_dir', type=str, required=True, help='Experiment dir')
        parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name')
        parser.add_argument('--run', type=str, required=True, help='Run name')

        self.add_commandline_arguments(parser)

        args = parser.parse_args()
        config = vars(args)

        return config

    def __enter__(self):
        """Used by the context manager (the "with" statement)
        
        Returns:
            Experiment
        """
        self.start_experiment()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Used by the context manger (the "with" statement).
        Finialize the experiment
        """
        if exc_type is None:
            self.finish_experiment()
        else:
            print('Experiment failed')

    def start_experiment(self):
        """Called when experiment is created 
        """
        print(f'Experiment started: {self.run_dir}')

    def finish_experiment(self):
        """
        Called when experiment has finished. Saves the configuration to a json file in the experiment dir
        """
        print(f'Experiment finished: {self.run_dir}')

        # save config
        
        save_json(self.config, os.path.join(self.run_dir, Experiment.CONFIG_FILENAME))

    def add_commandline_arguments(self, parser):
        """Abstract method to be defined in the child classes to add command line arguments
        
        Arguments:
            parser {argparse.ArgumentParser} -- Argument parser from the argparse module
        """
        pass

    def run(self):
        """Abstract method to be defined in the child classes. Should be called to run the experiemnt
        """
        pass


class ConfigFileExperiment(Experiment): 
    def __init__(self, config_name=None): 
        self._parse_args()
        self.config_name = config_name
        self.experiment_name = self.config['experiment_name']
        self.experiment_dir = self.config['experiment_dir']
        self.run_name = self.config['run']
        self.use_temp_dir = self.config['use_temp_dir']
        self.copy_model_checkpoint = self.config['copy_model_checkpoint']
        self.run_dir = os.path.join(self.experiment_dir, self.experiment_name, "run_" + str(self.run_name))

        self.working_dir = self.run_dir
        if self.use_temp_dir: 
            self.working_dir = os.path.join(self.config['temp_dir'], self.experiment_name, "run_" + str(self.run_name))

        self.setup()

        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.working_dir, exist_ok=True)

    def _parse_args(self): 
        arg_parser = argparse.ArgumentParser()
        required = arg_parser.add_argument_group('required_arguments')
        required.add_argument('-c', '--config', help='JSON config file.', type=argparse.FileType('r'), required=True)
        args = arg_parser.parse_args()
        self.config = json.load(args.config)
    
    def start_experiment(self):
        print(f'Experiment started!\nRun Directory -> {self.run_dir}')
        if self.use_temp_dir: 
            print(f'Using Temp Directory -> {self.working_dir}')

    def finish_experiment(self):
        if self.use_temp_dir: 
            print(f'Moving experiment results from Temp Directory to Run Directory...')
            if self.copy_model_checkpoint:
                shutil.copytree(self.working_dir, self.run_dir, dirs_exist_ok=True)
            else:
                shutil.copytree(self.working_dir, self.run_dir, ignore=shutil.ignore_patterns('*.pt','*.onnx'), dirs_exist_ok=True)

        print(f'Experiment Finished!\nRun Directory -> {self.run_dir}')

    def setup(self): 
        '''
        Read config file and initalize any member variables needed for the experiment run.
        To be defined by child classes. 
        '''
        pass

class BatchConfigFileExperiment(ConfigFileExperiment):
    @property
    @abstractmethod
    def pipeline(self): 
        '''
        Ordered list of (experiment_class, config_name) to run. 
        '''
    
    def run(self): 
        for exp, config_name in self.pipeline: 
            if self.config[config_name]['run_exp']:
                run_obj = exp(config_name = config_name)
                print(f'\nStarted Subexperiment -> {exp.__name__}...')
                run_obj.run()
                print(f'Finished Subexperiment -> {exp.__name__}!')
            else:
                print(f'\nSkipping Subexperiment -> {exp.__name__}...')

def update_nested_dict_with_nested_dict(dict_to_update: Dict, dict_with_updates: Dict): 
    for k, v in dict_with_updates.items(): 
        if k in dict_to_update: 
            if isinstance(v, dict):
                update_nested_dict_with_nested_dict(dict_to_update[k], v)
            else: 
                dict_to_update[k] = v
        else: 
            dict_to_update[k] = v
    
    return dict_to_update

def exps_from_config_diffs(config_diffs_path, generated_configs_dir=None, run_exps=False): 
    with open(config_diffs_path, "r") as config_diffs_file:
        config_diffs_dict = json.load(config_diffs_file)
        configs_and_metadata = config_diffs_dict['configs']
        curr_config_and_metadata = configs_and_metadata[0]

        if generated_configs_dir is not None: 
            os.makedirs(generated_configs_dir, exist_ok=True)

        for i in range(len(configs_and_metadata)): 
            curr_config_and_metadata = update_nested_dict_with_nested_dict(curr_config_and_metadata, configs_and_metadata[i])
            exp_script = curr_config_and_metadata['experiment_script']
            config = curr_config_and_metadata['config']
            
            if generated_configs_dir is not None: 
                config_name = config['run'] + '.json'
                config_file_path = os.path.join(generated_configs_dir, config_name)
                with open(config_file_path, "w") as config_file: 
                    json.dump(config, config_file, indent=4)
            
            if run_exps:
                command = f"python {os.path.abspath(exp_script)} --config {os.path.abspath(config_file_path)}" 
                p = subprocess.Popen(command)
                p.wait()