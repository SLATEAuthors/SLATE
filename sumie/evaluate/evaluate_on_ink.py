from sumie.data.annotation_utils import parse_annotations_for_text, merge_non_task_annotations
from sumie.data.sample_and_split import load, save
from sumie.utils.experiment import ConfigFileExperiment

from collections import defaultdict
from typing import List
from networkx.algorithms.bipartite.generators import complete_bipartite_graph
from networkx.algorithms.bipartite.matching import minimum_weight_full_matching
import pandas as pd
import numpy as np
import segeval
import argparse
import json
import os
import glob
import re

LABEL_MAP = {
            '<a0>' : '<a0>',
            '<a1>' : '<a1>',
            '<a2>' : '<a1>',
            '<a3>' : '<a1>',
            '<a4>' : '<a0>'
        }

class wr_to_ids:
    '''
    Used to map whitespace delimited tokens in a writing region to unique ids. 
    The usage of this encoder is NOT idempotent. A fresh instance should be created after a partial/full encoding. 
    ''' 
    def __init__(self, wr_text: str):
        words = wr_text.split()
        self.word_to_ids = defaultdict(list)
        for i in range(len(words)):
            self.word_to_ids[words[i]].append(i)
        self.word_to_ids_counter = defaultdict(int)
    
    def __call__(self, word: str): 
        '''
        Encode a word to the corresponding id in the writing region. 
        '''
        encoding = self.word_to_ids[word][self.word_to_ids_counter[word]]
        self.word_to_ids_counter[word] = (self.word_to_ids_counter[word] + 1) % len(self.word_to_ids[word])
        return encoding
    
    def encode_list_of_sentences(self, sentences: List[str]) -> List[list]:
        '''
        Batch encoding for a list of sentences.
        '''
        encodings = list()
        for sentence in sentences: 
            words = sentence.split()
            encodings.append([self.__call__(word) for word in words])
        return encodings
            

def jaccard_index(tokens_A: list, tokens_B: list) -> float: 
    '''
    Returns jaccard index between two lists of tokens.
    '''
    A = set(tokens_A)
    B = set(tokens_B)
    return len(A.intersection(B))/len(A.union(B))

def recall(tp, fn): 
    if (tp + fn) == 0:
        return np.nan
    return tp/(tp + fn)

def precision(tp, fp): 
    if (tp + fp) == 0:
        return np.nan
    return tp/(tp + fp)

def f1(tp, fp, fn): 
    r = recall(tp, fn)
    p = precision(tp, fp)
    if (r + p) == 0:
        return np.nan
    return (2*r*p)/(r + p)

def accuracy(tp, fp, tn, fn):
    if (tp + fp + tn + fn) == 0:
        return np.nan
    return (tp + tn)/(tp + fp + tn + fn)

def get_matching(sentences_A: List[list], sentences_B: List[list], metric=jaccard_index, threshold=0.25) -> dict: 
    '''
    Get best bipartite matching between lists A and B that maximizes sum of pairwise metric.
    If |A| > |B|, ties are broken based on order (first come first match).
    If |A| < |B|, remaining elements from B are not matched. 
    Provided metric function must have range [0, 1].
    '''
    if(len(sentences_A) == 0 or len(sentences_B) == 0): 
        return {}

    epsilon = 1e-10

    #Create metric weighted bipartite graph
    G = complete_bipartite_graph(len(sentences_A), len(sentences_B))
    for i in range(len(sentences_A)): 
        for j in range(len(sentences_B)):
            distance = 1 - metric(sentences_A[i], sentences_B[j])
            G[i][j + len(sentences_A)]['weight'] = distance + j*epsilon

    #Compute matching
    nodes_A = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0} 
    matching = minimum_weight_full_matching(G, top_nodes=nodes_A)

    processed_matching = {}
   
    #Prune dummy edges from the matching.
    matching_keys = list(matching.keys())
    for i in range(min(len(sentences_A), len(sentences_B))): 
        k = matching_keys[i]
        match = matching[k]
        if(G[k][match]['weight'] < 1 - threshold): 
            processed_matching[k] = match - len(sentences_A)
    
    return processed_matching

def aggregate_task_metrics(df: pd.DataFrame) -> dict: 
    mandatory_metrics = {
        'true_positives' : 0, 
        'false_positives' : 0, 
        'true_negatives' : 0,
        'false_negatives': 0
    }
    categories = ['<a0>', '<a1>', '<a2>', '<a3>', '<a4>']

    for category in categories:
            mandatory_metrics['{}_predicted'.format(category)] = 0
            mandatory_metrics['{}_actual'.format(category)] = 0
    
    for _, row in df['task_metrics'].iteritems():
        for metric in mandatory_metrics.keys(): 
            mandatory_metrics[metric] += row[metric]

    tp = mandatory_metrics['true_positives']
    fp = mandatory_metrics['false_positives']
    tn = mandatory_metrics['true_negatives']
    fn = mandatory_metrics['false_negatives']

    computed_metrics = {
        'task_recall': recall(tp, fn),
        'task_precision': precision(tp, fp),
        'task_f1' : f1(tp, fp, fn),
        'non-task_recall': recall(tn, fp), 
        'non-task_precision': precision(tn, fn), 
        'non-task_f1' : f1(tn, fn, fp),
        'accuracy': accuracy(tp, fp, tn, fn)
    }

    for category in categories:
        predicted = mandatory_metrics['{}_predicted'.format(category)]
        actual = mandatory_metrics['{}_actual'.format(category)]
        computed_metrics['{}_recall'.format(category)] = predicted/actual if actual > 0 else np.nan
    
    return pd.Series({'task_metrics' : {**computed_metrics, **mandatory_metrics}})

def aggregate_segmentation_metrics(df : pd.DataFrame, seg_level) -> dict:
    col_name = '{}_seg_metrics'.format(seg_level)
    metrics = list(df[col_name].iloc[0].keys())
    seg_metrics = dict.fromkeys(metrics, 0)
    num_rows = len(df)

    if seg_level == 'true_positive': 
        num_tp_rows = 0
        has_tps = False

        for _, row in df.iterrows(): 
            if row['task_metrics']['true_positives'] > 0: 
                has_tps = True
                num_tp_rows += 1
            else: 
                has_tps = False
            
            for metric in metrics: 
                seg_metrics[metric] += row[col_name][metric] if has_tps else 0
            
        for metric in metrics: 
            seg_metrics[metric] = seg_metrics[metric]/num_tp_rows if num_tp_rows > 0 else np.nan
    else: 
        for _, row in df[col_name].iteritems():
            for metric in metrics: 
                seg_metrics[metric] += (1/num_rows)*row[metric]

    return pd.Series({col_name : seg_metrics})

def get_task_matching(wr_text: str, predicted_annotation: str, ground_truth_annotation: str): 

    gt_sentences, gt_labels = parse_annotations_for_text(ground_truth_annotation)
    p_sentences, p_labels = parse_annotations_for_text(predicted_annotation)

    gt_encodings = wr_to_ids(wr_text).encode_list_of_sentences(gt_sentences)
    p_encodings = wr_to_ids(wr_text).encode_list_of_sentences(p_sentences)
    
    #Choose to only include tasks sentences in predicted set
    p_task_encodings = [p_encodings[i] for i in range(len(p_encodings)) if LABEL_MAP[p_labels[i]] == '<a1>']

    matching = get_matching(p_task_encodings, gt_encodings)

    return matching, gt_sentences, gt_labels, p_sentences, p_labels

def task_matching_to_string(matching: dict, p_sentences : List[str], p_labels: List[str], gt_sentences : List[str], gt_labels: List[str]):
    str_matching = {}
    true_positives = {}
    false_positives = {}
    true_negatives = []
    false_negatives = []
    
    p_tasks = [(p_sentences[i], p_labels[i]) for i in range(len(p_sentences)) if LABEL_MAP[p_labels[i]] == '<a1>']

    for k, v in matching.items(): 
        str_matching['<s>' + p_tasks[k][0] + p_tasks[k][1]] = '<s>' + gt_sentences[v] + gt_labels[v]
        if LABEL_MAP[gt_labels[v]] == '<a1>':
            true_positives['<s>' + p_tasks[k][0] + p_tasks[k][1]] = '<s>' + gt_sentences[v] + gt_labels[v]
        else:  
            false_positives['<s>' + p_tasks[k][0] + p_tasks[k][1]] = '<s>' + gt_sentences[v] + gt_labels[v]
    
    matched_gt_keys = list(matching.values())
    for i in range(len(gt_sentences)): 
        if LABEL_MAP[gt_labels[i]] == '<a1>' and i not in matched_gt_keys: 
            false_negatives.append(gt_sentences[i])
        elif LABEL_MAP[gt_labels[i]] == '<a0>' and i not in matched_gt_keys: 
            true_negatives.append(gt_sentences[i])
    
    return str_matching, true_positives, false_positives, true_negatives, false_negatives

def get_task_metrics(wr_text: str, predicted_annotation: str, ground_truth_annotation: str, \
    metrics=[
        'true_positives', 
        'false_positives', 
        'true_negatives',
        'false_negatives',
        'task_recall',
        'task_f1',
        'task_precision', 
        'non-task_recall', 
        'non-task_precision',
        'non-task_f1', 
        'accuracy']): 

        '''
        Description: 
        Computes task metrics for predicted annotation given ground truth annotations and whitespace delimited writing region text. 
        predicted_annotation must have task delimiters at a sentence level but non-task delimiters may be at a region level as shown in the example below. 

        Example input: 
        wr_text = 'To Do list Meet w/ Ruth cut the grass see I up the team roster read the case studies 7 Josh Turn left on ABC street'
        predicted_annotation = '<s>To Do list<a0><s>Meet w/<a1><s>Ruth cut the grass<a1><s>see I up the team<a3><s>roster read the case studies 7 Josh Turn left on ABC street<a0>'
        ground_truth_annotation = '<s>To Do list<a0><s>Meet w/ Ruth<a1><s>cut the grass<a1><s>see I up the team roster<a3><s>read the case studies 7 Josh<a1><s>Turn left on ABC street<a4>'
        
        Expected output: 
        {'true_positives': 3, 'false_positives': 0, 'true_negatives': 2, 'false_negatives': 1, ...}
        '''
        matching, gt_sentences, gt_labels, p_sentences, p_labels = get_task_matching(wr_text, predicted_annotation, ground_truth_annotation)

        tp, fp = 0, 0
        tn, fn = sum(1 if LABEL_MAP[label] == '<a0>' else 0 for label in gt_labels), sum(1 if LABEL_MAP[label] == '<a1>' else 0 for label in gt_labels)
        
        category_counts = {}
        for category in LABEL_MAP.keys():
            category_counts['{}_predicted'.format(category)] = 0
            category_counts['{}_actual'.format(category)] = sum(1 if l == category else 0 for l in gt_labels)

        category_counts['<a0>_predicted'] = sum(1 if l == '<a0>' else 0 for l in gt_labels)
        category_counts['<a4>_predicted'] = sum(1 if l == '<a4>' else 0 for l in gt_labels)


        for i in matching.keys():
            match = matching[i]
            if LABEL_MAP[gt_labels[match]] == '<a1>': 
                tp +=1
                fn -=1
                category_counts['{}_predicted'.format(gt_labels[match])] += 1

            else: 
                fp +=1
                tn -=1
                category_counts['{}_predicted'.format(gt_labels[match])] -= 1


        mandatory_metrics = {**category_counts, **{
            'true_positives' : tp, 
            'false_positives' : fp, 
            'true_negatives' : tn,
            'false_negatives': fn,
        }}
        
        computed_metrics = {**mandatory_metrics, **{
            'task_recall': recall(tp, fn),
            'task_precision': precision(tp, fp),
            'task_f1' : f1(tp, fp, fn),
            'non-task_recall': recall(tn, fp), 
            'non-task_precision': precision(tn, fn), 
            'non-task_f1' : f1(tn, fn, fp),
            'accuracy': accuracy(tp, fp, tn, fn)
        }}

        requested_metrics = {k: computed_metrics[k] for k in metrics}

        return ({**requested_metrics, **mandatory_metrics}, *task_matching_to_string(matching, p_sentences, p_labels, gt_sentences, gt_labels))

def interpret_bed(bed):
    '''
    Take a tuple containing three types of boundary edits and return their numbers.
    '''
    additions      = len(bed[0])
    substitutions  = len(bed[1])
    transpositions = len(bed[2])

    return (additions, substitutions, transpositions)

def get_true_positive_annotations(wr_text: str, predicted_annotation: str, ground_truth_annotation: str):
    matching, gt_sentences, gt_labels, p_sentences, p_labels = get_task_matching(wr_text, predicted_annotation, ground_truth_annotation)
    gt_matches = list(matching.values())
    
    gt_true_positive_annotations = ''
    for i in range(len(gt_sentences)):
        s = gt_sentences[i]
        if i in gt_matches and LABEL_MAP[gt_labels[i]] == '<a1>': 
            s = '<s>' + s + '<a1>'
        else: 
            s = '<s>' + s + '<a0>' 
        gt_true_positive_annotations += s

    _, true_positives, _, _, _ = task_matching_to_string(matching, p_sentences, p_labels, gt_sentences, gt_labels)
    
    p_true_positive_annotations = predicted_annotation
    p_true_positive_annotations = p_true_positive_annotations.replace('<a1>', '<a0>').replace('<a2>', '<a0>').replace('<a3>', '<a0>')
    for s in true_positives.keys():
        s_flipped = s.replace('<a1>', '<a0>').replace('<a2>', '<a0>').replace('<a3>', '<a0>')
        p_true_positive_annotations = p_true_positive_annotations.replace(s_flipped, s)

    predicted_annotation = merge_non_task_annotations(p_true_positive_annotations)
    ground_truth_annotation = merge_non_task_annotations(gt_true_positive_annotations)

    return predicted_annotation, ground_truth_annotation

def get_segmentation_metrics(wr_text: str, predicted_annotation: str, ground_truth_annotation: str, task_metrics: dict, seg_level: str,\
    metrics= [
        'boundary_similarity',
        'pk',
        'miss(es)', 
        'sub(s)', 
        'near']):
        
        #Add dummy non-tasks to the start and end of the writing region to handle cases where first or last sentence is a true positive.
        # This is to make sure that both boundaries of a true positive sentence are counted in seg metric calculations.  
        wr_text = 'padding ' + wr_text + ' padding'
        predicted_annotation = '<s>padding<a0>' + predicted_annotation + '<s>padding<a0>'
        ground_truth_annotation = '<s>padding<a0>' + ground_truth_annotation + '<s>padding<a0>'

        if(seg_level == 'region'):
            predicted_annotation = merge_non_task_annotations(predicted_annotation)
            ground_truth_annotation = merge_non_task_annotations(ground_truth_annotation)
        
        elif(seg_level == 'true_positive'):
            if task_metrics['true_positives'] == 0: 
                return dict.fromkeys(metrics, np.nan)
            predicted_annotation, ground_truth_annotation = get_true_positive_annotations(wr_text, predicted_annotation, ground_truth_annotation)

        gt_sentences, gt_labels = parse_annotations_for_text(ground_truth_annotation)
        p_sentences, p_labels = parse_annotations_for_text(predicted_annotation)
        ground_truth_segmentation = [len(s.split()) for s in gt_sentences]
        predicted_segmentation = [len(s.split()) for s in p_sentences]
  
        B, Pk, misses, subs, near = None, None, None, None, None
        if(ground_truth_segmentation == predicted_segmentation): 
            B, Pk, misses, subs, near = 1, 0, 0, 0, 0
        else: 
            B = float(segeval.boundary_similarity(ground_truth_segmentation, predicted_segmentation))
            Pk = float(segeval.pk(ground_truth_segmentation, predicted_segmentation))
            misses, subs, near = interpret_bed(segeval.boundary_edit_distance(segeval.boundary_string_from_masses(ground_truth_segmentation), 
                segeval.boundary_string_from_masses(predicted_segmentation))) 
        
        computed_metrics = {
            'boundary_similarity' : B,
            'pk' : Pk,
            'miss(es)' : misses, 
            'sub(s)': subs, 
            'near': near
        }

        return {k: computed_metrics[k] for k in metrics}

def evaluate_df(df: pd.DataFrame, task_metrics, seg_metrics, seg_levels): 
    df['task_metrics'], df['task_matches'], df['true_positives'], df['false_positives'], df['true_negatives'], df['false_negatives']= \
        zip(*df[['wr_text', 'predicted_annotation', 'ground_truth_annotation']].apply(lambda x: get_task_metrics(*x, metrics=task_metrics), axis=1))
    for level in seg_levels: 
        df['{}_seg_metrics'.format(level)] = df[['wr_text', 'predicted_annotation', 'ground_truth_annotation', 'task_metrics']]\
            .apply(lambda x: get_segmentation_metrics(*x, level, metrics=seg_metrics), axis=1)
    return df


class Evaluate(ConfigFileExperiment): 
    def __init__(self, config_name = 'eval_config', *args, **kwargs):
        super().__init__(config_name = config_name, *args, **kwargs)

    def setup(self): 
        eval_config = self.config[self.config_name]
        self.write_dynamic_file_extension = eval_config['write_dynamic_file_extension']
        self.task_metrics = eval_config['task_metrics']
        self.seg_metrics = eval_config['seg_metrics']
        self.seg_levels = eval_config['seg_levels']
        self.experiment_root = "leaderboard/"
        self.write_to_leaderboard = eval_config['write_to_leaderboard']
        self.leaderboard_name = eval_config['leaderboard_name']
        self.leaderboard_file_name = eval_config['leaderboard_file_name'] if 'leaderboard_file_name' in eval_config else "leaderboard.csv"


    def run(self): 
        inference_file_paths = glob.glob(os.path.join(self.working_dir, "inference*"))
        for inference_file_path in inference_file_paths:
            print(f'Evaluating {os.path.basename(inference_file_path)}...')
            self.prediction_df = load(inference_file_path)
            file_extension = ""
            leaderboard_name = self.leaderboard_name
            if self.write_dynamic_file_extension:
                m = re.search('inference(.+?).csv', os.path.basename(inference_file_path))
                if m:
                    file_extension = m.group(1)
                    leaderboard_name = self.leaderboard_name + file_extension
                else:
                    raise AttributeError('No file found with prefix inference for evaluation!')
            
            #writing region level metrics
            wr_evaluated_df = evaluate_df(self.prediction_df, self.task_metrics, self.seg_metrics, self.seg_levels)

            #whiteboard level metrics
            wh_metrics_dict = {}
            wh_metrics_dict['task_metrics'] = wr_evaluated_df.groupby('wh_id').apply(aggregate_task_metrics)['task_metrics']
            for seg_level in self.seg_levels: 
                wh_metrics_dict['{}_seg_metrics'.format(seg_level)] = wr_evaluated_df.groupby('wh_id').apply(aggregate_segmentation_metrics, seg_level=seg_level)['{}_seg_metrics'.format(seg_level)]
            wh_evaluated_df = pd.DataFrame(wh_metrics_dict).reset_index()

            #global aggregations
            global_metrics = {'config': self.config}
            global_metrics['task_metrics'] = aggregate_task_metrics(wr_evaluated_df).iloc[0]
            for seg_level in self.seg_levels: 
                global_metrics['{}_seg_metrics'.format(seg_level)] = aggregate_segmentation_metrics(wr_evaluated_df, seg_level=seg_level).iloc[0]

            #save results
            save(wr_evaluated_df, os.path.join(self.working_dir, 'wr_metrics{}.csv'.format(file_extension)))
            save(wh_evaluated_df, os.path.join(self.working_dir, 'wh_metrics{}.csv'.format(file_extension)))

            with open(os.path.join(self.working_dir, 'experiment_summary{}.json'.format(file_extension)), 'w') as result_file: 
                json.dump(global_metrics, result_file, indent=4)
        
            if self.write_to_leaderboard:
                leaderboard_path = os.path.join(self.experiment_root, self.leaderboard_file_name)
                leaderboard_df = None
                leaderboard_columns = [
                    "experiment",
                    "run_dir",
                    "config", 
                    "task_recall",
                    "task_precision",
                    "task_f1",
                    "non-task_recall",
                    "non-task_precision",
                    "non-task_f1",
                    "accuracy",
                    "<a0>_recall",
                    "<a1>_recall",
                    "<a2>_recall",
                    "<a3>_recall",
                    "<a4>_recall",
                    "true_positives",
                    "false_positives",
                    "true_negatives",
                    "false_negatives",
                    "<a0>_predicted",
                    "<a0>_actual",
                    "<a1>_predicted",
                    "<a1>_actual",
                    "<a2>_predicted",
                    "<a2>_actual",
                    "<a3>_predicted",
                    "<a3>_actual",
                    "<a4>_predicted",
                    "<a4>_actual", 
                    "true_positive_boundary_similarity",
                    "true_positive_pk",
                    "true_positive_miss(es)",
                    "true_positive_sub(s)",
                    "true_positive_near",
                    "sentence_boundary_similarity",
                    "sentence_pk",
                    "sentence_miss(es)",
                    "sentence_sub(s)",
                    "sentence_near",
                    "region_boundary_similarity",
                    "region_pk",
                    "region_miss(es)",
                    "region_sub(s)",
                    "region_near"
                ]

                if os.path.exists(leaderboard_path) and os.path.isfile(leaderboard_path): 
                    leaderboard_df = load(leaderboard_path)
                else: 
                    leaderboard_df = pd.DataFrame(columns = leaderboard_columns)
                
                new_row_index = len(leaderboard_df)
                leaderboard_df.loc[new_row_index, 'experiment'] = leaderboard_name
                leaderboard_df.loc[new_row_index, 'run_dir'] = self.run_dir
                leaderboard_df.loc[new_row_index, 'config'] = json.dumps(self.config)
                
                #task metrics
                for k, v in global_metrics['task_metrics'].items(): 
                    leaderboard_df.loc[new_row_index, '{}'.format(k)] = v

                #seg_metrics
                for seg_level in self.seg_levels:
                    seg_key = '{}_seg_metrics'.format(seg_level)
                    for k,v in global_metrics[seg_key].items(): 
                        leaderboard_df.loc[new_row_index, '{}_{}'.format(seg_level, k)] = v
                
                save(leaderboard_df, leaderboard_path)

if __name__ == '__main__':
    with Evaluate() as exp: 
        exp.run()