import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from typing import List, Tuple
import blingfire as bf
import os
import numpy as np
from abc import ABC, abstractmethod

class Tokenization(): 
    def __init__(self, texts: List[str], token_ids: List[List[int]], starts: List[List[int]], ends: List[List[int]]): 
        self._tokens = None
        self._char_to_tok_map = None
        self._token_ids = token_ids
        self._starts = starts
        self._ends = ends
        self._texts = texts
    
    def tokens(self) -> List[List[str]]:
        if self._tokens is not None: 
            return self._tokens
        
        tok_lists = list()
        for i in range(len(self._texts)): 
            starts = self._starts[i]
            ends = self._ends[i]
            text = self._texts[i]
            tok_lists.append([text[starts[j] : ends[j] + 1] for j in range(len(starts))])

        self._tokens = tok_lists

        return tok_lists
    
    def char_idx_to_token_idx(self, seq_idx: int, char_idx: int) -> int:
        if self._tokens is None: 
            self._tokens = self.tokens()
        
        if self._char_to_tok_map is None: 
            char_to_tok_map = []
            for s in range(len(self._texts)):
                char_to_tok_map.append([])
                for t in range(len(self._tokens[s])):
                    # s->sequence id, t->token index in sequence
                    token = self._tokens[s][t] 
                    for _ in range(len(token)): 
                        char_to_tok_map[s].append(t)
            self._char_to_tok_map = char_to_tok_map
        try:
            return self._char_to_tok_map[seq_idx][char_idx]
        except:
            print(self._char_to_tok_map)
            print(seq_idx)
            print(char_idx)
            print(self._tokens)
            raise Exception()
        

    def token_idx_to_token_str(self, seq_idx: int, tok_idx: int) -> str:
        return self.tokens()[seq_idx][tok_idx]

    def char_idx_to_token_str(self, seq_idx: int, char_idx: int) -> str: 
        return self.token_idx_to_token_str(seq_idx, self.char_idx_to_token_idx(seq_idx, char_idx))

    def token_idx_to_token_span(self, seq_idx: int, tok_idx: int) -> Tuple[int]:
        return (self._starts[seq_idx][tok_idx], self._ends[seq_idx][tok_idx] + 1)

    def char_idx_to_token_span(self, seq_idx: int, char_idx: int) -> Tuple[int]: 
        return self.token_idx_to_token_span(seq_idx, self.char_idx_to_token_idx(seq_idx, char_idx))
        
class BaseTokenizer(ABC):
    @abstractmethod
    def start_token(self) -> str: 
        pass 

    @abstractmethod
    def end_token(self) -> str: 
        pass

    @abstractmethod
    def add_special_toks(self, text: str) -> str:
        pass

    @abstractmethod
    def __call__(self, text_list : List[str]) -> dict: 
        pass
    
    @abstractmethod
    def get_tokenization(self, text_list: List[str]) -> Tokenization:
        pass


class RobertaTokenizer(BaseTokenizer):
        ''' 
        Tokenizer class for distilroberta built using the blingfire library. 
        ''' 
        def __init__(self): 
            self.tokenizer = bf.load_model(os.path.join(os.path.dirname(bf.__file__), "roberta.bin"))

        def start_token(self) -> str: 
            return '<s>'

        def end_token(self) -> str: 
            return '</s>'

        def add_special_toks(self, text: str) -> str: 
            return self.start_token() + text + self.end_token()
        
        def __call__(self, text_list : List[str], device='cpu', max_len=1000):
            max_len = max(len(t) for t in text_list) if not max_len else max_len
            text_list = [self.add_special_toks(text) for text in text_list]
            #Get token ids. Skip the first one since blingfire adds an unnecessary prefix space when tokenizing for roberta. 
            input_ids = [torch.tensor(bf.text_to_ids(self.tokenizer, text, max_len, unk=3, no_padding=True)[1:].astype(np.int64)) 
                for text in text_list]

            attention_mask = [torch.ones_like(seq) for seq in input_ids]
            
            input_ids = pad_sequence(input_ids, batch_first=True).to(device)
            attention_mask = pad_sequence(attention_mask, batch_first=True).to(device)

            return {'input_ids' : input_ids, 'attention_mask': attention_mask}
        
        def get_tokenization(self, text_list: List[str], max_len=1000) -> Tokenization: 
            max_len = max(len(t) for t in text_list) if not max_len else max_len
            text_list = [self.add_special_toks(text) for text in text_list]
            utf_text_list = [t.encode('utf-8') for t in text_list]
            arrs = [bf.utf8text_to_ids_with_offsets(self.tokenizer, text, max_len, unk=3, no_padding=True)
                for text in utf_text_list]
            #Skip the first one array elems below since blingfire adds an unnecessary prefix space when tokenizing for roberta. 
            token_ids = [a[0][1:].astype(np.int64) for a in arrs]
            starts = [a[1][1:].astype(np.int64) for a in arrs]
            ends = [a[2][1:].astype(np.int64) for a in arrs]
            return Tokenization(text_list, token_ids, starts, ends)

def get_tokenizer_from_name(tokenizer_name: str) -> BaseTokenizer: 
    tokenizer_map = {
        'roberta' : RobertaTokenizer
    }

    return tokenizer_map[tokenizer_name]()