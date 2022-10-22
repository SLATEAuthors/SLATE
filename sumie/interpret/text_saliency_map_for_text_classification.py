from sumie.utils.tokenization_utils import get_tokenizer_from_name
from sumie.utils.sequence_labeling_utils import get_labeling_scheme_from_name, SequenceLabelingBatch
from sumie.data.annotation_utils import get_raw_wr_text_from_annotation, insert_linebreaks_for_annotation
from sumie.models.roberta_clf import RobertaWithClfHead
from transformers import AutoModelForTokenClassification, RobertaModel, AutoTokenizer
from textualheatmap import TextualHeatmap
import torch
from torch.nn.functional import one_hot
import torch.autograd
import shap
import scipy as sp


def get_normalized_gradient_norm_for_binary_classification(model, tokenized_input, device, grad_type='one_hot'):
    embedding_matrix = model.roberta.embeddings.word_embeddings
    vocab_size = embedding_matrix.num_embeddings
    one_hots = one_hot(tokenized_input['input_ids'][0], vocab_size).float()
    one_hots.requires_grad = True
    one_hots.retain_grad()
    embeddings = one_hots @ embedding_matrix.weight.clone()
    if grad_type == 'embeddings': 
        embeddings.retain_grad()
    model.eval()
    model.zero_grad()
    output = model(inputs_embeds=embeddings.unsqueeze(dim=0),
       attention_mask=tokenized_input['attention_mask'])
    logits = output.squeeze()
    logits.max().backward(retain_graph=True)
    gradients = one_hots.grad.data if grad_type == 'one_hot' else embeddings.grad.data
    gradients_norms = torch.norm(gradients, dim=1)
    normalized_grad_norms = gradients_norms/gradients_norms.max()
    return logits.argmax().item(), normalized_grad_norms.tolist()

def get_normalized_shapley_scores_for_binary_classification(model, tokenizer, tokenized_input, device, sentence):
    model.eval()
    output = model(input_ids=tokenized_input['input_ids'],
       attention_mask=tokenized_input['attention_mask'])
    model.zero_grad()

    def prepare_inputs(batch_sentences, padding=True, return_tensors="pt"):
        inputs = tokenizer(batch_sentences.tolist(), padding=padding, return_tensors=return_tensors)
        return inputs

    def f(batch):
        inputs = prepare_inputs(batch)
        inputs = inputs.to(device)
        logits = model(**inputs)
        prob = torch.nn.Softmax(dim=1)(logits).detach().cpu().numpy()
        scores = sp.special.logit(prob)
        return scores

    masker = shap.maskers.Text(tokenizer, mask_token="...", collapse_mask_token=True)
    explainer = shap.Explainer(f, masker)
    shap_values = explainer([sentence])
    label = output.squeeze().argmax().item()
    values = shap_values.values[0,:,label]
    values = (values - values.min())/(values.max() - values.min())
    
    return label, values

def get_token_saliency_map(annotated_input : str, sentence_seg_model, task_classification_model, tokenizer, labeling_scheme, device, grad_type='one_hot', backend='grad'): 
    torch.enable_grad()
    sentence_seg_model.eval()
    task_classification_model.eval()
    model_input = SequenceLabelingBatch([annotated_input], labeling_scheme, tokenizer)
    wr_text_with_line_breaks = get_raw_wr_text_from_annotation(annotated_input)
    wr_text = wr_text_with_line_breaks.replace(' </>', '')

    sentence_seg_predicted_labels = sentence_seg_model(input_ids=model_input.token_ids_list.to(device),
        attention_mask=model_input.attention_mask_list.to(device))['logits'][0].argmax(axis=1).detach().cpu().numpy().tolist()
    sentences = labeling_scheme.extract_str_entities(wr_text_with_line_breaks, sentence_seg_predicted_labels, tokenizer)

    tokens = ['<s>']
    for i in range(len(sentences)):
        input_tok = tokenizer.get_tokenization([sentences[i]])
        sentence_tokens = input_tok.tokens()[0][1:-1]
        if i!=0:
            sentence_tokens[0] = ' ' + sentence_tokens[0]
        tokens += sentence_tokens
    tokens += ['</s>']
    num_tokens = len(tokens)

    #print(sentences)
    saliency_scores = []
    str_labels = []
    num_tokens_accounted_for = 0
    zeros = [0]*len(tokens)
    #wr bos saliency scores
    saliency_scores.append(zeros)
    for i in range(len(sentences)):
        clf_input = tokenizer([sentences[i]], device=device)

        if backend == 'grad':
            label, sentence_saliency_map = get_normalized_gradient_norm_for_binary_classification(task_classification_model, clf_input, device, grad_type)
        elif backend == 'shap':
            hf_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
            label, sentence_saliency_map = get_normalized_shapley_scores_for_binary_classification(task_classification_model, hf_tokenizer, clf_input, device, sentences[i])
        #prune out bos and eos tokens
        sentence_saliency_map = sentence_saliency_map[1:-1]

        wr_saliency_map = zeros.copy()
        #account for wr bos token
        if i == 0: 
            num_tokens_accounted_for += 1
            str_labels.append('-')

        wr_saliency_map[num_tokens_accounted_for : num_tokens_accounted_for + len(sentence_saliency_map)] = sentence_saliency_map
        sentence_start_label = 'T' if label==1 else 'N'
        sentence_labels = [sentence_start_label] + ['-']*(len(sentence_saliency_map) - 1)
        
        str_labels += sentence_labels
        #print(str_labels)
        num_tokens_accounted_for += len(sentence_saliency_map)

        #account for wr eos token 
        if i == len(sentences) - 1: 
            str_labels.append('-')

        for _ in range(len(sentence_saliency_map)):
            saliency_scores.append(wr_saliency_map)

    #wr eos saliency scores
    saliency_scores.append(zeros)
    
    #str_labels = labeling_scheme.int_to_str_labels(logits.argmax(dim=1).tolist())
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