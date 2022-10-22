import re
import pandas as pd
import numpy as np
from typing import List
from ast import literal_eval 

START_TOKEN_LIST = ['<s>']
END_TOKEN_LIST = ['<a0>', '<a1>', '<a2>', '<a3>', '<a4>']

def validate_annotation(text: str):
    text = text.strip() 
    is_correct = text[:3] in START_TOKEN_LIST \
        and text[-4:] in END_TOKEN_LIST \
        and sum(text.count(st) for st in START_TOKEN_LIST) \
            == sum(text.count(et) for et in END_TOKEN_LIST)
    return is_correct

def validate_annotations(df: pd.DataFrame):
    return df['annotate'].apply(validate_annotation).all()

def parse_annotations_for_text(text: str):
    pattern = "({})(.*?)({})".format('|'.join(START_TOKEN_LIST), '|'.join(END_TOKEN_LIST))
    matches = [s for s in re.findall(pattern, text)]
    sentences = [s[1].strip() for s in matches]
    labels = [s[2] for s in matches]
    return (sentences, labels)

def parse_annotations_for_df(df : pd.DataFrame):
    df = df.copy()
    df['sentences'] = df['annotate'].apply(lambda x : parse_annotations_for_text(x)[0])
    df['labels'] = df['annotate'].apply(lambda x : parse_annotations_for_text(x)[1])
    df['num_sentences'] = df['sentences'].apply(len)
    for et in END_TOKEN_LIST:
        df['num_{}'.format(et)] = df['labels'].apply(lambda x : sum(1 if elem == et else 0 for elem in x))
    return df

def parse_action_and_non_action_items_for_df(df: pd.DataFrame, action_item_token = ['<a1>', '<a2>', '<a3>'], non_action_item_token = ['<a0>', '<a4>']):
    df = parse_annotations_for_df(df)
    action_items, non_action_items = [], [] 
    
    for _,row in df.iterrows():
        sentences = row["sentences"]
        labels = row["labels"]
        for sentence,label in zip(sentences, labels):
            if label in action_item_token:
                action_items.append(sentence)
            else:
                non_action_items.append(sentence)

    return action_items, non_action_items

def merge_non_task_annotations(annotated_text : str): 
    sentences, labels = parse_annotations_for_text(annotated_text.replace('<a4>', '<a0>'))
    groupings = []
    current_grouping = [0]
    previous = labels[0]
    for i in range(1, len(labels)):
        label = labels[i] 
        if previous == '<a0>' and label == '<a0>': 
                current_grouping.append(i)
                continue
        else: 
            groupings.append(current_grouping)
            current_grouping = [i]
        previous = label
    groupings.append(current_grouping)

    new_annotation = ''
    for grouping in groupings: 
        region = '<s>'
        if(len(grouping) > 1): 
            region += ' '.join([sentences[index] for index in grouping])
            region += '<a0>'
        else: 
            region += sentences[grouping[0]] + labels[grouping[0]]
        new_annotation += region
         
    return new_annotation

def make_task_annotations_binary(annotated_text : str): 
    return annotated_text.replace('<a3>', '<a1>').replace('<a2>', '<a1>').replace('<a4>', '<a0>')

def get_raw_wr_text_from_annotation(annotated_text: str): 
    wr_sentences, _ = parse_annotations_for_text(annotated_text)
    return ' '.join(wr_sentences)

def insert_linebreaks_for_annotation(annotated_text: str, wr_lines: List[str], line_break_token: str ='</>') -> str:
    # insert spaces for bos and eos tokens
    annotated_text = annotated_text.replace('<s>', '<s> ')
    for et in END_TOKEN_LIST:
        annotated_text = annotated_text.replace(et, ' '+et+' ')
    words_in_annotated_text = annotated_text.split()
    annotated_text_with_line_breaks = ''
    words_ptr = 0
    for i in range(len(wr_lines)):
        line = wr_lines[i]
        line_ptr = 0
        while line_ptr < len(line.split()):
            curr_word = words_in_annotated_text[words_ptr]
            if curr_word in (START_TOKEN_LIST+END_TOKEN_LIST):
                annotated_text_with_line_breaks += curr_word
                words_ptr += 1
            else:
                annotated_text_with_line_breaks += curr_word + ' '
                words_ptr += 1
                line_ptr += 1
    
        annotated_text_with_line_breaks += line_break_token + ' '
             
    annotated_text_with_line_breaks += words_in_annotated_text[words_ptr]

    for et in END_TOKEN_LIST:
        annotated_text_with_line_breaks = annotated_text_with_line_breaks.replace(' '+et, et)
    
    return annotated_text_with_line_breaks

def insert_bullets_for_annotation(annotated_text: str, wr_lines: List[str], bullet_mask: List[int], 
    line_break_token: str = '</>', bullet_token: str ='<.>') -> str:
    
    # insert spaces for bos and eos tokens
    annotated_text = annotated_text.replace('<s>', '<s> ')
    for et in END_TOKEN_LIST:
        annotated_text = annotated_text.replace(et, ' '+et+' ')
    words_in_annotated_text = annotated_text.split()
    annotated_text_with_bullets = ''
    words_ptr = 0

    for i in range(len(wr_lines)):
        line = wr_lines[i]

        line_ptr = 0
        while line_ptr < len(line.split()):
            curr_word = words_in_annotated_text[words_ptr]

            if curr_word in (START_TOKEN_LIST + END_TOKEN_LIST + [line_break_token]):
                annotated_text_with_bullets += curr_word
                words_ptr += 1

                if curr_word == line_break_token: 
                    annotated_text_with_bullets += ' '

                if curr_word in START_TOKEN_LIST and bullet_mask[i]:
                    annotated_text_with_bullets += bullet_token + ' '

            else:
                annotated_text_with_bullets += curr_word + ' '
                words_ptr += 1
                line_ptr += 1
        
    if words_in_annotated_text[words_ptr] == line_break_token: 
        annotated_text_with_bullets += line_break_token + ' '
        words_ptr += 1
            
    #handle eos token
    annotated_text_with_bullets += words_in_annotated_text[words_ptr]

    for et in END_TOKEN_LIST:
        annotated_text_with_bullets = annotated_text_with_bullets.replace(' '+et, et)
    
    return annotated_text_with_bullets

def create_wr_context_feature_for_sentence_with_labels(annotated_text: str, sentence_start_token: str = '<st>', sentence_end_token: str = '</st>'):
    sentences, labels = parse_annotations_for_text(annotated_text)
    sentences_with_context = []
    for i in range(len(sentences)):
        sentence = sentence_start_token + ' ' + sentences[i] + ' ' + sentence_end_token
        before_context = ' '.join(sentences[0:i])
        after_context = ' '.join(sentences[i+1:])
        sentence_with_context = before_context + ' ' + sentence + ' ' + after_context
        sentences_with_context.append(sentence_with_context.strip())
    
    return sentences_with_context, labels

def str_label_to_int(labels: List[str], scheme: str = 'default'):
    transformed_labels = []
    for label in labels:
        if scheme == 'default':
            if label in ['<a1>', '<a2>', '<a3>']:
                transformed_labels.append(1)
            elif label in ['<a0>', '<a4>']:
                transformed_labels.append(0)
        if scheme == 'only_a0_and_a1':
            if label in ['<a1>']:
                transformed_labels.append(1)
            elif label in ['<a0>']:
                transformed_labels.append(0)
            elif label in ['<a2>', '<a3>', '<a4>']:
                transformed_labels.append(-1)
    return transformed_labels

def filter_bullet_chars_for_text(text: str): 
    for st in START_TOKEN_LIST: 
        text = text.replace(st, st + ' ')
    for et in END_TOKEN_LIST: 
        text = text.replace(et, ' ' + et)

    pattern_exclude = ' *[^ ]*[Ã|Â|°|Â|Ë|Å|Ž|º|â|Î][^ ]* '
    matches = [s for s in re.findall(pattern_exclude, text)]
    num_matches = len(matches)

    for _ in range(num_matches):
        match = re.findall(pattern_exclude, text)[0] 
        text = text.replace(match, ' ')
    
    for st in START_TOKEN_LIST: 
        text = text.replace(st + ' ', st)
    for et in END_TOKEN_LIST: 
        text = text.replace(' ' + et, et)
    
    return text

def filter_bullet_chars_for_df(df: pd.DataFrame):
    df = df.copy()
    df_new = pd.DataFrame(columns=df.columns)
    for i, row in df.iterrows(): 
        annotate = filter_bullet_chars_for_text(row['annotate'])
        wr_lines = [filter_bullet_chars_for_text(' ' + s + ' ').strip() for s in literal_eval(row['wr_lines'])]
        wr_sentences = [filter_bullet_chars_for_text(' ' + s + ' ').strip() for s in literal_eval(row['wr_sentences'])]
        # filter to non-empty lines and sentences
        wr_lines = str([s for s in wr_lines if s])
        wr_sentences = str([s for s in wr_sentences if s])
        df_new.loc[i] = row
        df_new.loc[i, 'annotate'] = annotate
        df_new.loc[i, 'wr_lines'] = wr_lines
        df_new.loc[i, 'wr_sentences'] = wr_sentences
        
    return df_new
