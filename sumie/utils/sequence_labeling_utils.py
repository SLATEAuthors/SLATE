from sumie.utils.tokenization_utils import RobertaTokenizer, BaseTokenizer
from sumie.utils.sequence_labeling_schemes import *
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Union

class SequenceLabelingBatch(): 
    def __getitem__(self, item): 
        return getattr(self, item)
    
    def __init__(self, annotated_wrs: List[str], labeling_scheme: Union[str, BaseSequenceLabelingScheme], tokenizer: BaseTokenizer): 
        self.annotated_wrs = annotated_wrs
        self.labeling_scheme = labeling_scheme
        self.tokenizer = tokenizer

        if(isinstance(labeling_scheme, str)): 
            labeling_scheme = get_labeling_scheme_from_name(labeling_scheme)

        tokens_list, token_ids_list, labels_list, label_ids_list = list(), list(), list(), list() 

        for annotated_wr in annotated_wrs:
            try:
                tokens, token_ids, labels, label_ids = labeling_scheme.get_token_tags_for_annotation(annotated_wr, tokenizer)
            except:
                print(annotated_wr)
                raise Exception()
            tokens_list.append(tokens)
            token_ids_list.append(token_ids)
            labels_list.append(labels)
            label_ids_list.append(label_ids)

        token_ids_list = [torch.tensor(token_ids) for token_ids in token_ids_list]
        attention_mask_list = [torch.ones_like(token_ids) for token_ids in token_ids_list]

        token_ids_list = pad_sequence(token_ids_list, batch_first=True)
        attention_mask_list = pad_sequence(attention_mask_list, batch_first=True)
        max_seq_len = len(token_ids_list[0])
        label_ids_list = torch.tensor([label_ids + [labeling_scheme.str_to_int_label_map[labeling_scheme.padding_label]]*(max_seq_len - len(label_ids))
            for label_ids in label_ids_list])

        self.tokens_list = tokens_list
        self.token_ids_list = token_ids_list #[:, :512]
        self.labels_list = labels_list
        self.label_ids_list = label_ids_list #[:, :512]
        self.attention_mask_list = attention_mask_list #[:, :512]

class SequenceSplitHandler(): 
    def reset(self):
        self.batch_idx_to_splits_maps = {}
        self.num_splits = 0

    def _split_single_default(self, sequence: torch.Tensor, max_len: int, batch_index, stride: int=None):
        if not stride: 
            stride = max_len//2
        splits = sequence.unfold(0, max_len, stride)
        remaining_elems = len(sequence) - (splits.numel() - stride*(len(splits) - 1))
        if remaining_elems != 0: 
            remaining_split = torch.zeros(max_len).to(splits.device)
            remaining_split[:remaining_elems] = sequence[-remaining_elems:]
            splits = torch.cat((splits, remaining_split.unsqueeze(0)))

        self.batch_idx_to_splits_maps[self.curr_sequence_type][batch_index] = list(range(self.num_splits, self.num_splits + len(splits)))
        self.num_splits += len(splits)
        return splits
        

    def _split_single_input_ids(self, sequence: torch.Tensor, max_len: int, batch_index, stride: int=None): 
        splits = self._split_single_default(sequence, max_len - 2, batch_index, stride=stride)
        processed_splits = []
        
        bos_tensor = self.bos_token.clone().detach().to(splits.device).unsqueeze(0)
        eos_tensor = self.eos_token.clone().detach().to(splits.device).unsqueeze(0)
        padding_tensor = torch.zeros_like(bos_tensor)
        
        #handle first split
        first_split = torch.cat((splits[0], eos_tensor, padding_tensor)).unsqueeze(0)
        processed_splits.append(first_split)

        #handle middle splits
        for i in range(1, len(splits) - 1): 
            split = torch.cat((bos_tensor, splits[i], eos_tensor)).unsqueeze(0)
            processed_splits.append(split)

        #handle final split
        final_split = torch.cat((bos_tensor, splits[-1], padding_tensor)).unsqueeze(0)
        processed_splits.append(final_split)
        
        return torch.cat(processed_splits).int()

    def _split_single_attention_mask(self, sequence: torch.Tensor, max_len: int, batch_index, stride: int=None):
        splits = self._split_single_default(sequence, max_len - 2, batch_index, stride=stride)
        processed_splits = []

        active_tensor = torch.tensor(1).to(splits.device).unsqueeze(0)
        padding_tensor = torch.zeros_like(active_tensor)

        #handle first split
        first_split = torch.cat((splits[0], active_tensor, padding_tensor)).unsqueeze(0)
        processed_splits.append(first_split)

        #handle middle splits
        for i in range(1, len(splits) - 1): 
            split = torch.cat((active_tensor, splits[i], active_tensor)).unsqueeze(0)
            processed_splits.append(split)

        #handle final split
        final_split = torch.cat((active_tensor, splits[-1], padding_tensor)).unsqueeze(0)
        processed_splits.append(final_split)
        
        return torch.cat(processed_splits).int()

    def _split_single(self, sequence: torch.Tensor, max_len: int, batch_index, stride: int=None, sequence_type: str='default'):
        func_inputs = (sequence, max_len, batch_index, stride)
        split_fn_dict = {
            'default': self._split_single_default, 
            'input_ids': self._split_single_input_ids, 
            'attention_mask': self._split_single_attention_mask
        }        
        return split_fn_dict[sequence_type](*func_inputs) if sequence_type in split_fn_dict else sequence

    def _split_batch(self, batch: torch.Tensor, max_len: int, stride: int=None, sequence_type: str='default'): 
        self.curr_sequence_type = sequence_type
        self.batch_idx_to_splits_maps[sequence_type] = {}
        return torch.cat([self._split_single(batch[i], max_len, i, stride=stride, sequence_type=sequence_type) for i in range(len(batch))])
    
    def split_input_dict(self, input_dict: Dict, max_len: int, stride: int=None):
        self.reset()
        sample_attention_mask = input_dict['attention_mask'][0]
        
        if len(sample_attention_mask) <= max_len: 
            return input_dict

        sample_input_ids = input_dict['input_ids'][0]
        active_input_ids = sample_input_ids[sample_attention_mask.bool()]
        self.bos_token = active_input_ids[0]
        self.eos_token = active_input_ids[-1]
        self.pre_split_seq_len = len(sample_attention_mask)

        processed_input_dict = input_dict
        processed_input_dict['input_ids'] = self._split_batch(input_dict['input_ids'], max_len, stride, sequence_type='input_ids')
        processed_input_dict['attention_mask'] = self._split_batch(input_dict['attention_mask'], max_len, stride, sequence_type='attention_mask')

        return processed_input_dict

    def _merge_single_logits(self, batch: torch.Tensor, max_len: int, batch_index, stride: int=None): 
        if not stride: 
            stride = max_len//2

        splits = batch[self.batch_idx_to_splits_maps['input_ids'][batch_index][0]: self.batch_idx_to_splits_maps['input_ids'][batch_index][-1] + 1]
        processed_splits = []
        
        #handle first split
        processed_splits.append(splits[0][:-2][:stride])

        #handle middle splits
        for i in range(1, len(splits) - 1): 
            processed_splits.append(splits[i][1:-1][:stride])

        #handle final split 
        processed_splits.append(splits[-1][1:-1])

        return torch.cat(processed_splits)[:self.pre_split_seq_len].unsqueeze(0)

    def _merge_single(self, batch: torch.Tensor, max_len: int, batch_index, stride: int=None, sequence_type: str='default'):
        func_inputs = (batch, max_len - 2, batch_index, stride)
        merge_fn_dict = {
            'logits': self._merge_single_logits
        }
        return merge_fn_dict[sequence_type](*func_inputs)

    def _merge_batch(self, batch: torch.Tensor, max_len: int, stride: int=None, sequence_type: str='logits'):
        batch_idx_to_splits_map = self.batch_idx_to_splits_maps['input_ids']
        return torch.cat([self._merge_single(batch, max_len, i, stride=stride, sequence_type=sequence_type) for i in range(len(batch_idx_to_splits_map))])

    def merge_output_dict(self, output_dict: Dict, max_len: int, stride: int=None): 
        if len(self.batch_idx_to_splits_maps) == 0: 
            return output_dict

        processed_output_dict = output_dict
        processed_output_dict['logits'] = self._merge_batch(output_dict['logits'], max_len, stride, sequence_type='logits')
        
        return processed_output_dict

def get_labeling_scheme_from_name(labeling_scheme_name: str) -> BaseSequenceLabelingScheme: 
    return {
        'sentence_BI': SentenceBISequenceLabelingScheme,
        'sentence_BI_with_line_breaks': SentenceBISequenceLabelingSchemeWithLineBreaks,
        'sentence_BI_with_bullets': SentenceBISequenceLabelingSchemeWithBullets,
        'sentence_BI_with_line_breaks_and_bullets': SentenceBISequenceLabelingSchemeWithLineBreaksAndBullets,
        'task_BIO' : TaskBIOSequenceLabelingScheme, 
        'task_BIO_with_line_breaks': TaskBIOSequenceLabelingSchemeWithLineBreaks,
        'task_BIO_with_bullets': TaskBIOSequenceLabelingSchemeWithBullets,
        'task_BIO_with_line_breaks_and_bullets': TaskBIOSequenceLabelingSchemeWithLineBreaksAndBullets,
        'task_NTI': TaskNTISequenceLabelingScheme, 
        'task_NTI_with_line_breaks': TaskNTISequenceLabelingSchemeWithLineBreaks,
        'task_NTI_with_bullets': TaskNTISequenceLabelingSchemeWithBullets,
        'task_NTI_with_line_breaks_and_bullets': TaskNTISequenceLabelingSchemeWithLineBreaksAndBullets
        }[labeling_scheme_name]()

def get_token_tags_for_annotation(annotated_text: str, tokenizer, labeling_scheme='task_BIO'): 
    return get_labeling_scheme_from_name(labeling_scheme).get_token_tags_for_annotation(annotated_text, tokenizer)
            
# %%            
if __name__ == '__main__': 
    annotated_wr = '<s>To Do list<a0><s>Meet w/ Apurva<a1><s>cut the grass<a1><s>see I up the team roster<a3><s>read the case studies 7 Josh<a1><s>Turn left on ABC street<a4>'
    tokenizer = RobertaTokenizer()
    tokens, token_ids, tags, tag_ids = get_token_tags_for_annotation(annotated_wr, tokenizer, labeling_scheme='task_BIO')
    print(list(zip(tokens, token_ids, tags, tag_ids)))
    tokens, token_ids, tags, tag_ids = get_token_tags_for_annotation(annotated_wr, tokenizer, labeling_scheme='sentence_BI')
    print(list(zip(tokens, token_ids, tags, tag_ids)))

    batch = SequenceLabelingBatch([annotated_wr, annotated_wr + '<s>Hello world<a0>'], get_labeling_scheme_from_name('task_BIO'), tokenizer) 
    print(batch.token_ids_list)
    print(batch.attention_mask_list)
    print(batch.label_ids_list)

    