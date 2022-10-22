from sumie.utils.tokenization_utils import BaseTokenizer
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

class BaseSequenceLabelingScheme(ABC):
    '''
    Base class from which all sequence labeling schemes inherit. 
    '''

    @property
    @abstractmethod
    def name(self) -> str: 
        pass

    @property
    @abstractmethod
    def labels(self) -> List[str]: 
        pass

    @property
    @abstractmethod
    def str_to_int_label_map(self) -> Dict[str, int]:
        pass

    @property
    @abstractmethod
    def int_to_str_label_map(self) -> Dict[str, int]:
        pass

    @property
    @abstractmethod
    def padding_label(self) -> str: 
        '''
        Label assigned to padding tokens. 
        '''
        pass

    @abstractmethod
    def get_token_tags_for_annotation(self, annotated_text: str, tokenizer: BaseTokenizer) -> Tuple[List[str], List[int], List[str], List[int]]: 
        '''
        Given some annotated text and the tokenizer, return a tuple of (string tokens, token_ids, string token_labels, integer token labels).
        '''
        pass

    @abstractmethod
    def extract_str_entities(self, text: str, label_ids: List[int], tokenizer: BaseTokenizer) -> List[str]: 
        '''
        Given the input text and the sequence labeling model's token-level integer labels for this text, extract the desired string entities
        (e.g., returns a list of sentences extracted from text for a sentence segmentation labeling scheme).
        '''
        pass

    @abstractmethod
    def annotate_text(self, text: str, labels_ids: List[int], tokenizer: BaseTokenizer) -> str:
        '''
        Given the input text and the sequence labeling model's token-level integer labels for this text, annotate the text according to our 
        annotation scheme. 
        ''' 
        pass

    @abstractmethod
    def char_labels_to_word_label(self, label_ids: List[int]) -> int:
        '''
        Convert a list of char-level integer labels for a given word to a single label for the word. 
        '''
        pass

    def str_to_int_labels(self, label_list: List[str]) -> List[int]: 
        return [self.str_to_int_label_map[l] for l in label_list]
    
    def int_to_str_labels(self, label_ids: List[int]) -> List[str]:
        return [self.int_to_str_label_map[l] for l in label_ids]
    
    def aggregate_token_labels_to_words(self, text: str, label_ids: List[int], tokenizer: BaseTokenizer) -> List[int]:
        '''
        Takes an array/list of token-level integer labels and returns word-level integer labels. 
        '''
        tokenization = tokenizer.get_tokenization([text])
        tokens = tokenization.tokens()[0]
        # ignore padded labels
        label_ids = label_ids[:len(tokens)]
        # ignoring begin and end of sentence tokens
        tokens = tokens[1 : -1]
        label_ids = label_ids[1 : -1]
        word_level_labels = []
        current_word_labels = []
        for i, token in enumerate(tokens):
            if token[0] == ' ': 
                word_level_labels.append(self.char_labels_to_word_label(current_word_labels))
                current_word_labels = []
            
            label = label_ids[i]
            # explode token level labels to character level labels (motivation: longer tokens get bigger say in aggregation)
            current_word_labels += [label]*len(token.strip())
        word_level_labels.append(self.char_labels_to_word_label(current_word_labels))
        return word_level_labels

def filter_token_from_words_and_labels(words: List[str], word_labels: List[str], token: str):
        filtered_words, filtered_word_labels = [], []
        for i,word in enumerate(words):
            if word != token:
                filtered_words.append(word)
                filtered_word_labels.append(word_labels[i])
        return filtered_words, filtered_word_labels