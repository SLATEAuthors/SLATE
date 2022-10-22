from sumie.utils.sequence_labeling_schemes.base import BaseSequenceLabelingScheme, filter_token_from_words_and_labels
from sumie.utils.tokenization_utils import BaseTokenizer
from sumie.data.annotation_utils import parse_annotations_for_text, make_task_annotations_binary
from typing import List
from scipy.stats import mode
import re

class TaskBIOSequenceLabelingScheme(BaseSequenceLabelingScheme): 

    @property
    def name(self): 
        return 'task_BIO'
    
    @property
    def labels(self): 
        return ['B', 'I', 'O']
    
    @property
    def str_to_int_label_map(self): 
        return {'B' : 0, 'I' : 1, 'O' : 2}
    
    @property
    def int_to_str_label_map(self): 
        return {0 : 'B', 1 : 'I', 2: 'O'}
    
    @property
    def padding_label(self): 
        return 'O'
    
    def get_token_tags_for_annotation(self, annotated_text: str, tokenizer: BaseTokenizer): 
        annotated_text = make_task_annotations_binary(annotated_text)
        wr_sentences, labels = parse_annotations_for_text(annotated_text)
        wr_text = ' '.join(wr_sentences)
        
        tokenized_text = tokenizer.get_tokenization([wr_text])
        tokens = tokenized_text.tokens()[0]
        token_labels = ['O']*len(tokens)
        token_ids = [elem for elem in tokenized_text._token_ids[0]]

        char_idx = 0
        for i in range(len(labels)): 
            label = labels[i]

            if i == 0: 
                char_idx += len(tokenizer.start_token())
                if label == '<a0>': 
                    char_idx += len(wr_sentences[i])
                    continue

            if label == '<a0>': 
                char_idx += len(' ' + wr_sentences[i])
                continue

            label_token_idx_set = set()
            
            for _ in range(len(' ' + wr_sentences[i])):
                token_idx = tokenized_text.char_idx_to_token_idx(0, char_idx)
                label_token_idx_set.add(token_idx)
                char_idx += 1
            
            for num, token_idx in enumerate(sorted(label_token_idx_set)): 
                 token_labels[token_idx] = 'B' if num == 0 else 'I'
        
        return (tokens, token_ids, token_labels, self.str_to_int_labels(token_labels))

    def _extract_str_entities_from_words_and_labels(self, words: List[str], word_labels: List[str]):
        label_str = ''.join(word_labels)
        entity_spans = [m.span() for m in re.finditer('BI*', label_str)]
        entities = [' '.join(words[s[0]:s[1]]).strip() for s in entity_spans]
        return entities

    def _annotate_text_from_words_and_labels(self, words: List[str], word_labels: List[str]):
        label_str = ''.join(word_labels)
        entity_spans = [m.span() for m in re.finditer('BI*', label_str)]
        for span in entity_spans:
            words[span[0]] = '<s>' + words[span[0]]
            words[span[1] - 1] = words[span[1] - 1] + '<a1>'
        annotated_wr = ' '.join(words)
        annotated_wr = annotated_wr.replace('<a1> <s>', '<a1><s>').replace(' <s>', '<a0><s>').replace('<a1> ', '<a1><s>')
        
        # handles case when first sentence is not a task
        if(annotated_wr[:3] != '<s>'): 
            annotated_wr = '<s>' + annotated_wr

        # handles case when last sentence is not a task
        if(annotated_wr[-4:] != '<a1>'):
            annotated_wr = annotated_wr + '<a0>'
        
        return annotated_wr
    

    def extract_str_entities(self, text: str,  label_ids: List[int], tokenizer: BaseTokenizer): 
        words = text.split()
        word_labels = self.int_to_str_labels(self.aggregate_token_labels_to_words(text, label_ids, tokenizer))
        return self._extract_str_entities_from_words_and_labels(words, word_labels)

    def annotate_text(self, text: str, label_ids: List[int], tokenizer: BaseTokenizer):
        words = text.split()
        word_labels = self.int_to_str_labels(self.aggregate_token_labels_to_words(text, label_ids, tokenizer))
        return self._annotate_text_from_words_and_labels(words, word_labels)


    def char_labels_to_word_label(self, label_ids: List[int]) -> int:
        if 0 in label_ids: 
            return 0
        return mode(label_ids)[0][0]

class TaskBIOSequenceLabelingSchemeWithLineBreaks(TaskBIOSequenceLabelingScheme):
    @property
    def name(self) -> str: 
        return 'task_BIO_with_line_breaks'

    def extract_str_entities(self, text: str,  label_ids: List[int], tokenizer: BaseTokenizer, line_break_token: str='</>'): 
        words = text.split()
        word_labels = self.int_to_str_labels(self.aggregate_token_labels_to_words(text, label_ids, tokenizer))
        words, word_labels = filter_token_from_words_and_labels(words, word_labels, token=line_break_token)
        return self._extract_str_entities_from_words_and_labels(words, word_labels)

    def annotate_text(self, text: str, label_ids: List[int], tokenizer: BaseTokenizer, line_break_token: str='</>'):
        words = text.split()
        word_labels = self.int_to_str_labels(self.aggregate_token_labels_to_words(text, label_ids, tokenizer))
        words, word_labels = filter_token_from_words_and_labels(words, word_labels, token=line_break_token)
        return self._annotate_text_from_words_and_labels(words, word_labels)
    
class TaskBIOSequenceLabelingSchemeWithBullets(TaskBIOSequenceLabelingScheme):
    @property
    def name(self) -> str: 
        return 'task_BIO_with_bullets'

    def get_token_tags_for_annotation(self, annotated_text: str, tokenizer: BaseTokenizer): 
        annotated_text = make_task_annotations_binary(annotated_text)
        wr_sentences, labels = parse_annotations_for_text(annotated_text)
        wr_text = ' '.join(wr_sentences)
        
        tokenized_text = tokenizer.get_tokenization([wr_text])
        tokens = tokenized_text.tokens()[0]
        token_labels = ['O']*len(tokens)
        token_ids = [elem for elem in tokenized_text._token_ids[0]]

        char_idx = 0
        for i in range(len(labels)): 
            label = labels[i]

            if i == 0: 
                char_idx += len(tokenizer.start_token())
                if label == '<a0>': 
                    char_idx += len(wr_sentences[i])
                    continue

            if label == '<a0>': 
                char_idx += len(' ' + wr_sentences[i])
                continue

            label_token_idx_set = set()
            
            #do not label bullet as start of sentence
            starts_with_bullet = wr_sentences[i].startswith('<.>')
            bullet_offset = 3 if i == 0 else 4
            if starts_with_bullet:
                char_idx += bullet_offset

            for _ in range(len(('' if i==0 else ' ') + wr_sentences[i]) - (bullet_offset if starts_with_bullet else 0)):
                token_idx = tokenized_text.char_idx_to_token_idx(0, char_idx)
                label_token_idx_set.add(token_idx)
                char_idx += 1
            
            for num, token_idx in enumerate(sorted(label_token_idx_set)): 
                 token_labels[token_idx] = 'B' if num == 0 else 'I'
        
        return (tokens, token_ids, token_labels, self.str_to_int_labels(token_labels))

    def extract_str_entities(self, text: str,  label_ids: List[int], tokenizer: BaseTokenizer, bullet_token: str='<.>'): 
        words = text.split()
        word_labels = self.int_to_str_labels(self.aggregate_token_labels_to_words(text, label_ids, tokenizer))
        words, word_labels = filter_token_from_words_and_labels(words, word_labels, token=bullet_token)
        return self._extract_str_entities_from_words_and_labels(words, word_labels)

    def annotate_text(self, text: str, label_ids: List[int], tokenizer: BaseTokenizer, bullet_token: str='<.>'):
        words = text.split()
        word_labels = self.int_to_str_labels(self.aggregate_token_labels_to_words(text, label_ids, tokenizer))
        words, word_labels = filter_token_from_words_and_labels(words, word_labels, token=bullet_token)
        return self._annotate_text_from_words_and_labels(words, word_labels)

class TaskBIOSequenceLabelingSchemeWithLineBreaksAndBullets(TaskBIOSequenceLabelingSchemeWithBullets):
    @property
    def name(self) -> str: 
        return 'task_BIO_with_line_breaks_and_bullets'

    def extract_str_entities(self, text: str,  label_ids: List[int], tokenizer: BaseTokenizer,
        line_break_token: str='</>', bullet_token: str='<.>'):

        words = text.split()
        word_labels = self.int_to_str_labels(self.aggregate_token_labels_to_words(text, label_ids, tokenizer))
        words, word_labels = filter_token_from_words_and_labels(words, word_labels, token=bullet_token)
        words, word_labels = filter_token_from_words_and_labels(words, word_labels, token=line_break_token)
        return self._extract_str_entities_from_words_and_labels(words, word_labels)

    def annotate_text(self, text: str, label_ids: List[int], tokenizer: BaseTokenizer, 
        line_break_token: str='</>', bullet_token: str='<.>'): 

        words = text.split()
        word_labels = self.int_to_str_labels(self.aggregate_token_labels_to_words(text, label_ids, tokenizer))
        words, word_labels = filter_token_from_words_and_labels(words, word_labels, token=bullet_token)
        words, word_labels = filter_token_from_words_and_labels(words, word_labels, token=line_break_token)
        return self._annotate_text_from_words_and_labels(words, word_labels)