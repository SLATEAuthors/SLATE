from sumie.utils.sequence_labeling_schemes.base import BaseSequenceLabelingScheme, filter_token_from_words_and_labels
from sumie.utils.tokenization_utils import BaseTokenizer
from sumie.data.annotation_utils import parse_annotations_for_text, make_task_annotations_binary
from typing import List
from collections import Counter
import re

class TaskNTISequenceLabelingScheme(BaseSequenceLabelingScheme): 

    @property
    def name(self): 
        return 'task_NTI'
    
    @property
    def labels(self): 
        return ['N', 'T', 'I']
    
    @property
    def str_to_int_label_map(self): 
        return {'N' : 0, 'T' : 1, 'I' : 2}
    
    @property
    def int_to_str_label_map(self): 
        return {0 : 'N', 1 : 'T', 2: 'I'}
    
    @property
    def padding_label(self): 
        return 'I'
    
    def get_token_tags_for_annotation(self, annotated_text: str, tokenizer: BaseTokenizer): 
        annotated_text = make_task_annotations_binary(annotated_text)
        wr_sentences, labels = parse_annotations_for_text(annotated_text)
        wr_text = ' '.join(wr_sentences)
        
        tokenized_text = tokenizer.get_tokenization([wr_text])
        tokens = tokenized_text.tokens()[0]
        token_labels = ['I']*len(tokens)
        token_ids = [elem for elem in tokenized_text._token_ids[0]]

        char_idx = 0
        for i in range(len(labels)): 
            label = labels[i]

            if i == 0: 
                char_idx += len(tokenizer.start_token())

            label_token_idx_set = set()
            
            for _ in range(len(('' if i==0 else ' ') + wr_sentences[i])):
                token_idx = tokenized_text.char_idx_to_token_idx(0, char_idx)
                label_token_idx_set.add(token_idx)
                char_idx += 1
            
            bos_label = 'N' if label == '<a0>' else 'T'
            for num, token_idx in enumerate(sorted(label_token_idx_set)): 
                 token_labels[token_idx] = bos_label if num == 0 else 'I'
        
        return (tokens, token_ids, token_labels, self.str_to_int_labels(token_labels))
    
    def _extract_str_entities_from_words_and_labels(self, words: List[str], word_labels: List[str], entity_type='tasks'):
        if word_labels[0] not in ['N', 'T']: 
            word_labels[0] = 'N'
        label_str = ''.join(word_labels)
        task_spans = [m.span() for m in re.finditer('TI*', label_str)]
        tasks = [' '.join(words[s[0]:s[1]]).strip() for s in task_spans]
        non_task_spans = [m.span() for m in re.finditer('NI*', label_str)]
        non_tasks = [' '.join(words[s[0]:s[1]]).strip() for s in non_task_spans]
        sentence_spans = [m.span() for m in re.finditer('[N|T]I*', label_str)]
        sentences = [' '.join(words[s[0]:s[1]]).strip() for s in sentence_spans]

        if entity_type == 'tasks': 
            return tasks
        elif entity_type == 'non_tasks':
            return non_tasks
        elif entity_type == 'sentences': 
            return sentences
        elif entity_type == 'all':
            return (tasks, non_tasks, sentences)
        
        raise ValueError('Provided enitity_type is undefined!')

    def _annotate_text_from_words_and_labels(self, words: List[str], word_labels: List[str]):
        if word_labels[0] not in ['N', 'T']: 
            word_labels[0] = 'N'
        label_str = ''.join(word_labels)
        task_spans = [m.span() for m in re.finditer('TI*', label_str)]
        for span in task_spans:
            words[span[0]] = '<s>' + words[span[0]]
            words[span[1] - 1] = words[span[1] - 1] + '<a1>'

        non_task_spans = [m.span() for m in re.finditer('NI*', label_str)]
        for span in non_task_spans:
            words[span[0]] = '<s>' + words[span[0]]
            words[span[1] - 1] = words[span[1] - 1] + '<a0>'

        annotated_wr = ' '.join(words)
        annotated_wr = annotated_wr.replace('<a1> <s>', '<a1><s>').replace('<a0> <s>', '<a0><s>')
        
        return annotated_wr
        
    def extract_str_entities(self, text: str,  label_ids: List[int], tokenizer: BaseTokenizer, entity_type='tasks'): 
        words = text.split()
        word_labels = self.int_to_str_labels(self.aggregate_token_labels_to_words(text, label_ids, tokenizer))
        return self._extract_str_entities_from_words_and_labels(words, word_labels, entity_type=entity_type)

    def annotate_text(self, text: str, label_ids: List[int], tokenizer: BaseTokenizer):
        words = text.split()
        word_labels = self.int_to_str_labels(self.aggregate_token_labels_to_words(text, label_ids, tokenizer))
        return self._annotate_text_from_words_and_labels(words, word_labels)

    def char_labels_to_word_label(self, label_ids: List[int]) -> int:
        N_present = 0 in label_ids
        T_present = 1 in label_ids

        if N_present and not T_present: 
            return 0
        elif T_present and not N_present: 
            return 1
        elif not N_present and not T_present: 
            return 2
        else: 
            counts = Counter(label_ids)
            if counts[0] > counts[1]: 
                return 0
            else: 
                return 1

class TaskNTISequenceLabelingSchemeWithLineBreaks(TaskNTISequenceLabelingScheme):
    @property
    def name(self) -> str: 
        return 'task_NTI_with_line_breaks'
        
    def extract_str_entities(self, text: str,  label_ids: List[int], tokenizer: BaseTokenizer, entity_type='tasks', line_break_token: str='</>'): 
        words = text.split()
        word_labels = self.int_to_str_labels(self.aggregate_token_labels_to_words(text, label_ids, tokenizer))
        words, word_labels = filter_token_from_words_and_labels(words, word_labels, token=line_break_token)
        return self._extract_str_entities_from_words_and_labels(words, word_labels, entity_type=entity_type)

    def annotate_text(self, text: str, label_ids: List[int], tokenizer: BaseTokenizer, line_break_token: str='</>'):
        words = text.split()
        word_labels = self.int_to_str_labels(self.aggregate_token_labels_to_words(text, label_ids, tokenizer))
        words, word_labels = filter_token_from_words_and_labels(words, word_labels, token=line_break_token)
        return self._annotate_text_from_words_and_labels(words, word_labels)

class TaskNTISequenceLabelingSchemeWithBullets(TaskNTISequenceLabelingScheme):
    @property
    def name(self) -> str: 
        return 'task_NTI_with_bullets'
    
    def get_token_tags_for_annotation(self, annotated_text: str, tokenizer: BaseTokenizer): 
        annotated_text = make_task_annotations_binary(annotated_text)
        wr_sentences, labels = parse_annotations_for_text(annotated_text)
        wr_text = ' '.join(wr_sentences)
        
        tokenized_text = tokenizer.get_tokenization([wr_text])
        tokens = tokenized_text.tokens()[0]
        token_labels = ['I']*len(tokens)
        token_ids = [elem for elem in tokenized_text._token_ids[0]]

        char_idx = 0
        for i in range(len(labels)): 
            label = labels[i]

            if i == 0: 
                char_idx += len(tokenizer.start_token())

            label_token_idx_set = set()
            
            starts_with_bullet = wr_sentences[i].startswith('<.>')
            bullet_offset = 3 if i == 0 else 4
            if starts_with_bullet:
                char_idx += bullet_offset

            for _ in range(len(('' if i==0 else ' ') + wr_sentences[i]) - (bullet_offset if starts_with_bullet else 0)):
                token_idx = tokenized_text.char_idx_to_token_idx(0, char_idx)
                label_token_idx_set.add(token_idx)
                char_idx += 1
            
            bos_label = 'N' if label == '<a0>' else 'T'
            for num, token_idx in enumerate(sorted(label_token_idx_set)): 
                 token_labels[token_idx] = bos_label if num == 0 else 'I'

        return (tokens, token_ids, token_labels, self.str_to_int_labels(token_labels))

    def extract_str_entities(self, text: str,  label_ids: List[int], tokenizer: BaseTokenizer, entity_type='tasks', bullet_token: str='<.>'): 
        words = text.split()
        word_labels = self.int_to_str_labels(self.aggregate_token_labels_to_words(text, label_ids, tokenizer))
        words, word_labels = filter_token_from_words_and_labels(words, word_labels, token=bullet_token)
        return self._extract_str_entities_from_words_and_labels(words, word_labels, entity_type=entity_type)

    def annotate_text(self, text: str, label_ids: List[int], tokenizer: BaseTokenizer, bullet_token: str='<.>'):
        words = text.split()
        word_labels = self.int_to_str_labels(self.aggregate_token_labels_to_words(text, label_ids, tokenizer))
        words, word_labels = filter_token_from_words_and_labels(words, word_labels, token=bullet_token)
        return self._annotate_text_from_words_and_labels(words, word_labels)

class TaskNTISequenceLabelingSchemeWithLineBreaksAndBullets(TaskNTISequenceLabelingSchemeWithBullets):
    @property
    def name(self) -> str: 
        return 'task_NTI_with_line_breaks_and_bullets'

    def extract_str_entities(self, text: str,  label_ids: List[int], tokenizer: BaseTokenizer, entity_type='tasks', 
        line_break_token: str='</>', bullet_token: str='<.>'): 

        words = text.split()
        word_labels = self.int_to_str_labels(self.aggregate_token_labels_to_words(text, label_ids, tokenizer))
        words, word_labels = filter_token_from_words_and_labels(words, word_labels, token=bullet_token)
        words, word_labels = filter_token_from_words_and_labels(words, word_labels, token=line_break_token)
        return self._extract_str_entities_from_words_and_labels(words, word_labels, entity_type=entity_type)

    def annotate_text(self, text: str, label_ids: List[int], tokenizer: BaseTokenizer, 
        line_break_token: str='</>', bullet_token: str='<.>'):

        words = text.split()
        word_labels = self.int_to_str_labels(self.aggregate_token_labels_to_words(text, label_ids, tokenizer))
        words, word_labels = filter_token_from_words_and_labels(words, word_labels, token=bullet_token)
        words, word_labels = filter_token_from_words_and_labels(words, word_labels, token=line_break_token)
        return self._annotate_text_from_words_and_labels(words, word_labels)
