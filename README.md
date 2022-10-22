# SLATE
Official repository for SLATE: A Sequence Labeling Approach for Task Extraction from Free-form Content  

# Code Organization
```
├── docs                    <- Contains documentation such as the dataset Annotation Guide. 
├── configs                 <- Contains the configs used to run the paper experiments.
├── data                    <- Contains the train, test, and val datasets.
├── environment.yml         <- Environment file with all dependencies required to use the sumie package.
├── leaderboard             <- Leaderboard for experiment runs is written here.
├── setup.py                <- Setup script to install the sumie package.
├── sumie                   <- Python package for the SLATE paper.
│   ├── batch_experiments   <- End-to-end experiment scripts (pipelining train, predict, evaluate and export experiments.) 
│   ├── data                <- Dataset annotation and dataframe utils. 
│   ├── datasets            <- Dataset classes for the experiments. 
│   ├── evaluate            <- Evaluation experiment scripts.
│   ├── interpret           <- Utilities to visualize explantations of model predictions. 
│   ├── post_process        <- Post-process experiment scripts.
│   ├── predict             <- Predict/Inference experiment scripts.
│   ├── train               <- Train experiment scripts.
│   ├── utils               <- Various utils. 
├── README.md               <- Top-level README file with project description.
```

# Installation
1. Install miniconda/anaconda if you do not already have this installed. 
2. To build the sumie conda env, use the command: ```conda env create -f environment.yml```
3. Activate the sumie conda environment: ```conda activate sumie``` 
4. Install the sumie package in developer mode: ```pip install -e .``` 
5. If you have a machine with a GPU and cuda installed (recommended), please follow the follow the next two steps as well.  
    * Uninstall CPU versions of tensorflow and torch:  
      * ```pip uninstall torch``` 
      * ```pip uninstall tensorflow``` 
    * Install the pytorch version matching exactly your installed cuda version.

# Run Experiments from the Paper
1. Edit fields with values 'EDIT: ...' in the config files under the configs folder with your own local paths. 
2. Run the following script from the repo root: ```python run_paper_experiments.py```

# Dataset
The train, validation, and test datasets used in the SLATE paper can be found in data/train.csv, data/val.csv and data/test.csv respectively. 
The Annotate column contains text recognized from ink documents annotated according to the following:  
* \<s>sentence\<a0> : Non-task sentence (e.g., I love ink!) 
* \<s>sentence\<a1> : Task sentence (e.g., Schedule the code review meeting for tomorrow.)
* \<s>sentence\<a2> : Generic task (e.g., I will do it.)
* \<s>sentence\<a3> : Task sentence due to context.
* \<s>sentence\<a4> : Non-task sentence due to context.

In the SLATE paper, all except the \<a0> annotation tag are considered task sentences. 

The wr_lines column gives the list of document lines for the recognized text. For experiments that use line breaks, we insert them between these document lines. 
The line_list_item mask column is a list with the same length as wr_lines where a 1 represents that the line is a bulleted item and 0 represents that the line is not bulleted. 

wh_id is the id of the whiteboard/ink document that the text is from. 
wr_id is the id of the writing region in the document that the text belongs to. 

For more details on how the dataset was annotated, please refer to the Annotation Guide in the docs folder. 

# Note
This repo and its resources are protected by a patent and usage is permitted only for research purposes. 
