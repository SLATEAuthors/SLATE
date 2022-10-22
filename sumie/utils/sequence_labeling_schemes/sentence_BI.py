from sumie.utils.sequence_labeling_schemes.base import BaseSequenceLabelingScheme, filter_token_from_words_and_labels
from sumie.utils.tokenization_utils import BaseTokenizer
from sumie.data.annotation_utils import parse_annotations_for_text
from typing import List
from scipy.stats import mode
import re

class SentenceBISequenceLabelingScheme(BaseSequenceLabelingScheme): 

    @property
    def name(self): 
        return 'sentence_BI'
    
    @property
    def labels(self): 
        return ['B', 'I']
    
    @property
    def str_to_int_label_map(self): 
        return {'B' : 0, 'I' : 1}
    
    @property
    def int_to_str_label_map(self): 
        return {0 : 'B', 1 : 'I'}
    
    @property
    def padding_label(self):
        return 'I'

    def get_token_tags_for_annotation(self, annotated_text: str, tokenizer: BaseTokenizer): 
        wr_sentences, labels = parse_annotations_for_text(annotated_text)
        wr_text = ' '.join(wr_sentences)
        
        tokenized_text = tokenizer.get_tokenization([wr_text])
        tokens = tokenized_text.tokens()[0]
        token_labels = ['I']*len(tokens)
        token_ids = [elem for elem in tokenized_text._token_ids[0]]

        char_idx = 0
        for i in range(len(wr_sentences)):
            if i == 0: 
                char_idx += len(tokenizer.start_token())
            # token_idx is the postional index of the token in the writing region
            token_idx = tokenized_text.char_idx_to_token_idx(0, char_idx)
            token_labels[token_idx] = 'B'

            char_idx += len(wr_sentences[i]) if i == 0 else len(' ' + wr_sentences[i])

        return (tokens, token_ids, token_labels, self.str_to_int_labels(token_labels))

    def _extract_str_entities_from_words_and_labels(self, words: List[str], word_labels: List[str]):
        if word_labels[0] != 'B': 
            word_labels[0] = 'B'
        label_str = ''.join(word_labels)
        entity_spans = [m.span() for m in re.finditer('BI*', label_str)]
        entities = [' '.join(words[s[0]:s[1]]).strip() for s in entity_spans]
        return entities

    def _annotate_text_from_words_and_labels(self, words: List[str], word_labels: List[str]):
        if word_labels[0] != 'B': 
            word_labels[0] = 'B'
        label_str = ''.join(word_labels)
        entity_spans = [m.span() for m in re.finditer('BI*', label_str)]
        for span in entity_spans:
            words[span[0]] = '<s>' + words[span[0]]
            words[span[1] - 1] = words[span[1] - 1] + '<a0>'
        annotated_wr = ' '.join(words)
        annotated_wr = annotated_wr.replace('<a0> <s>', '<a0><s>')
        
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
        

class SentenceBISequenceLabelingSchemeWithLineBreaks(SentenceBISequenceLabelingScheme):
    @property
    def name(self) -> str: 
        return 'sentence_BI_with_line_breaks'

    def extract_str_entities(self, text: str,  label_ids: List[int], tokenizer: BaseTokenizer, line_break_token: str='</>'): 
        words = text.split()
        word_labels = self.int_to_str_labels(self.aggregate_token_labels_to_words(text, label_ids, tokenizer))
        words, word_labels = filter_token_from_words_and_labels(words, word_labels, token=line_break_token)
        return self._extract_str_entities_from_words_and_labels(words, word_labels)
    
    def annotate_text(self, text: str, label_ids: List[int], tokenizer: BaseTokenizer, line_break_token: str='</>'): 
        words = text.split()
        word_labels = self.int_to_str_labels(self.aggregate_token_labels_to_words(text, label_ids, tokenizer))
        words, word_labels = self.filter_token_from_words_and_labels(words, word_labels, token=line_break_token)
        return self._annotate_text_from_words_and_labels(words, word_labels)    

class SentenceBISequenceLabelingSchemeWithBullets(SentenceBISequenceLabelingScheme):
    @property
    def name(self) -> str: 
        return 'sentence_BI_with_bullets'
    
    def get_token_tags_for_annotation(self, annotated_text: str, tokenizer: BaseTokenizer): 
        wr_sentences, labels = parse_annotations_for_text(annotated_text)
        wr_text = ' '.join(wr_sentences)
        
        tokenized_text = tokenizer.get_tokenization([wr_text])
        tokens = tokenized_text.tokens()[0]
        token_labels = ['I']*len(tokens)
        token_ids = [elem for elem in tokenized_text._token_ids[0]]

        char_idx = 0
        for i in range(len(wr_sentences)):
            if i == 0: 
                char_idx += len(tokenizer.start_token())
            # token_idx is the postional index of the token in the writing region
            token_idx = tokenized_text.char_idx_to_token_idx(0, char_idx)
            
            #do not label bullet as start of sentence
            if(wr_sentences[i].startswith('<.>')): 
                token_idx = tokenized_text.char_idx_to_token_idx(0, char_idx + (3 if i == 0 else 4))
            
            token_labels[token_idx] = 'B'

            char_idx += len(wr_sentences[i]) if i == 0 else len(' ' + wr_sentences[i])

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

class SentenceBISequenceLabelingSchemeWithLineBreaksAndBullets(SentenceBISequenceLabelingScheme):
    @property
    def name(self) -> str: 
        return 'sentence_BI_with_line_breaks_and_bullets'

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