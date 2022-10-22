from sumie.utils.tokenization_utils import get_tokenizer_from_name
from sumie.utils.sequence_labeling_utils import get_labeling_scheme_from_name, SequenceLabelingBatch
from sumie.data.annotation_utils import get_raw_wr_text_from_annotation
from transformers import AutoModelForTokenClassification, AutoTokenizer
from textualheatmap import TextualHeatmap
import torch
from torch.nn.functional import one_hot
import torch.autograd
import numpy as np
import scipy as sp
import shap


def get_normalized_gradient_norm_for_token_id(token_num: int, model, tokenized_batch, device, grad_type='one_hot'):
    embedding_matrix = model.roberta.embeddings.word_embeddings
    vocab_size = embedding_matrix.num_embeddings
    one_hots = one_hot(tokenized_batch.token_ids_list.to(device)[0], vocab_size).float()
    one_hots.requires_grad = True
    one_hots.retain_grad()
    embeddings = one_hots @ embedding_matrix.weight.clone()
    if grad_type == 'embeddings': 
        embeddings.retain_grad()
    model.eval()
    model.zero_grad()
    output = model(inputs_embeds=embeddings.unsqueeze(dim=0),
        attention_mask=tokenized_batch.attention_mask_list.to(device))
    logits = output['logits'].squeeze()
    token_logits = logits[token_num].unsqueeze(dim=0)
    token_logits.max().backward(retain_graph=True)
    gradients = one_hots.grad.data if grad_type == 'one_hot' else embeddings.grad.data
    gradients_norms = torch.norm(gradients, dim=1)
    normalized_grad_norms = gradients_norms/gradients_norms.max()
    return normalized_grad_norms.tolist()
    
def get_normalized_shapley_scores_for_sequence_labeling(model, tokenizer, tokenized_input, device, sentence):
    model.eval()
    output = model(input_ids=tokenized_input.token_ids_list.to(device),
       attention_mask=tokenized_input.attention_mask_list.to(device))
    logits = output.logits
    logit_idx = torch.argmax(logits, dim=2).squeeze().cpu().numpy()
    model.zero_grad()

    def prepare_inputs(batch_sentences, padding=True, return_tensors="pt"):
        inputs = tokenizer(batch_sentences.tolist(), padding=padding, return_tensors=return_tensors)
        return inputs

    def slice_log_odds(scores, pos_ids):
        extracted_log_odds = []
        for i,pos_id in enumerate(pos_ids):
            sliced_log_odds = scores[:,i,pos_id]
            extracted_log_odds.append(sliced_log_odds)
        return np.vstack(extracted_log_odds).T

    def f(batch):
        inputs = prepare_inputs(batch)
        inputs = inputs.to(device)
        logits = model(**inputs).logits
        prob = torch.nn.Softmax(dim=2)(logits).detach().cpu().numpy()
        scores = sp.special.logit(prob)
        sliced_scores = slice_log_odds(scores, logit_idx)
        return sliced_scores

    masker = shap.maskers.Text(tokenizer, mask_token="auto", collapse_mask_token=False)
    explainer = shap.Explainer(f, masker)
    shap_values = explainer([sentence], max_evals=100)
    values = shap_values.values[0,:,:]
    values = (values - values.min(axis=0))/(values.max(axis=0) - values.min(axis=0))
    
    return values

def get_token_saliency_map(annotated_input : str, model, tokenizer, labeling_scheme, device, grad_type='one_hot', backend='grad'): 
    torch.enable_grad()
    model.eval()
    model_input = SequenceLabelingBatch([annotated_input], labeling_scheme, tokenizer)
    wr_text = get_raw_wr_text_from_annotation(annotated_input)
    tokenization = tokenizer.get_tokenization([wr_text])
    tokens = tokenization.tokens()[0]
    num_tokens = len(tokens)
    
    saliency_scores = []
    if backend == 'grad':
        for i in range(num_tokens):
            saliency_scores.append(get_normalized_gradient_norm_for_token_id(i, model, model_input, device, grad_type))
    elif backend == 'shap':
        hf_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        scores = get_normalized_shapley_scores_for_sequence_labeling(model, hf_tokenizer, model_input, device, wr_text)
        for i in range(scores.shape[-1]):
            saliency_scores.append(scores[:,i].tolist())
    
    
    output = model(input_ids=model_input.token_ids_list.to(device),
        attention_mask=model_input.attention_mask_list.to(device), 
        output_hidden_states=True)
    logits = output['logits'].squeeze()
    
    str_labels = labeling_scheme.int_to_str_labels(logits.argmax(dim=1).tolist())
    heatmap_data = [[], []]
    for i in range(num_tokens):
        heatmap_data[0].append({
            'token': tokens[i],
            'heat': saliency_scores[i]
        })

        heatmap_data[1].append({
            'token': str_labels[i]+ '  ', 
            'heat': [1 if j == i else 0 for j in range(num_tokens)]
        }) 
    
    heatmap = TextualHeatmap(
        width=600,
        facet_titles= ['wr_text', 'token_labels'], 
        show_meta=False,
        rotate_facet_titles=True
    )
    heatmap.set_data(heatmap_data)