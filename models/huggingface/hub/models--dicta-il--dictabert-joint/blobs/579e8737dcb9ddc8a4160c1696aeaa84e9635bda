from collections import OrderedDict
from operator import itemgetter
from transformers.utils import ModelOutput
import torch
from torch import nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast

ALL_POS = ['DET', 'NOUN', 'VERB', 'CCONJ', 'ADP', 'PRON', 'PUNCT', 'ADJ', 'ADV', 'SCONJ', 'NUM', 'PROPN', 'AUX', 'X', 'INTJ', 'SYM']
ALL_PREFIX_POS = ['SCONJ', 'DET', 'ADV', 'CCONJ', 'ADP', 'NUM']
ALL_SUFFIX_POS = ['none', 'ADP_PRON', 'PRON']
ALL_FEATURES = [
    ('Gender', ['none', 'Masc', 'Fem', 'Fem,Masc']),
    ('Number', ['none', 'Sing', 'Plur', 'Plur,Sing', 'Dual', 'Dual,Plur']),
    ('Person', ['none', '1', '2', '3', '1,2,3']),
    ('Tense', ['none', 'Past', 'Fut', 'Pres', 'Imp'])
]

@dataclass
class MorphLogitsOutput(ModelOutput):
    prefix_logits: torch.FloatTensor = None
    pos_logits: torch.FloatTensor = None
    features_logits: List[torch.FloatTensor] = None
    suffix_logits: torch.FloatTensor = None
    suffix_features_logits: List[torch.FloatTensor] = None

    def detach(self):
        return MorphLogitsOutput(self.prefix_logits.detach(), self.pos_logits.detach(), [logits.deatch() for logits in self.features_logits], self.suffix_logits.detach(), [logits.deatch() for logits in self.suffix_features_logits])


@dataclass
class MorphTaggingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[MorphLogitsOutput] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class MorphLabels(ModelOutput):
    prefix_labels: Optional[torch.FloatTensor] = None
    pos_labels: Optional[torch.FloatTensor] = None
    features_labels: Optional[List[torch.FloatTensor]] = None
    suffix_labels: Optional[torch.FloatTensor] = None
    suffix_features_labels: Optional[List[torch.FloatTensor]] = None

    def detach(self):
        return MorphLabels(self.prefix_labels.detach(), self.pos_labels.detach(), [labels.detach() for labels in self.features_labels], self.suffix_labels.detach(), [labels.detach() for labels in self.suffix_features_labels])
    
    def to(self, device):
        return MorphLabels(self.prefix_labels.to(device), self.pos_labels.to(device), [feat.to(device) for feat in self.features_labels], self.suffix_labels.to(device), [feat.to(device) for feat in self.suffix_features_labels])

class BertMorphTaggingHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.num_prefix_classes = len(ALL_PREFIX_POS)
        self.num_pos_classes = len(ALL_POS)
        self.num_suffix_classes = len(ALL_SUFFIX_POS)
        self.num_features_classes = list(map(len, map(itemgetter(1), ALL_FEATURES)))
        # we need a classifier for prefix cls and POS cls
        # the prefix will use BCEWithLogits for multiple labels cls
        self.prefix_cls = nn.Linear(config.hidden_size, self.num_prefix_classes)
        # and pos + feats will use good old cross entropy for single label
        self.pos_cls = nn.Linear(config.hidden_size, self.num_pos_classes)
        self.features_cls = nn.ModuleList([nn.Linear(config.hidden_size, len(features)) for _, features in ALL_FEATURES])
        # and suffix + feats will also be cross entropy
        self.suffix_cls = nn.Linear(config.hidden_size, self.num_suffix_classes)
        self.suffix_features_cls = nn.ModuleList([nn.Linear(config.hidden_size, len(features)) for _, features in ALL_FEATURES])

    def forward(
            self, 
            hidden_states: torch.Tensor,
            labels: Optional[MorphLabels] = None):
        # run each of the classifiers on the transformed output
        prefix_logits = self.prefix_cls(hidden_states)
        pos_logits = self.pos_cls(hidden_states)
        suffix_logits = self.suffix_cls(hidden_states)
        features_logits = [cls(hidden_states) for cls in self.features_cls]
        suffix_features_logits = [cls(hidden_states) for cls in self.suffix_features_cls]

        loss = None
        if labels is not None:
            # step 1: prefix labels loss
            loss_fct = nn.BCEWithLogitsLoss(weight=(labels.prefix_labels != -100).float())
            loss = loss_fct(prefix_logits, labels.prefix_labels)
            # step 2: pos labels loss
            loss_fct = nn.CrossEntropyLoss()
            loss += loss_fct(pos_logits.view(-1, self.num_pos_classes), labels.pos_labels.view(-1))
            # step 2b: features
            for feat_logits,feat_labels,num_features in zip(features_logits, labels.features_labels, self.num_features_classes):
                loss += loss_fct(feat_logits.view(-1, num_features), feat_labels.view(-1))
            # step 3: suffix logits loss
            loss += loss_fct(suffix_logits.view(-1, self.num_suffix_classes), labels.suffix_labels.view(-1))
            # step 3b: suffix features
            for feat_logits,feat_labels,num_features in zip(suffix_features_logits, labels.suffix_features_labels, self.num_features_classes):
                loss += loss_fct(feat_logits.view(-1, num_features), feat_labels.view(-1))

        return loss, MorphLogitsOutput(prefix_logits, pos_logits, features_logits, suffix_logits, suffix_features_logits)

class BertForMorphTagging(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.morph = BertMorphTaggingHead(config)
        
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[MorphLabels] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        bert_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = bert_outputs[0]
        hidden_states = self.dropout(hidden_states)

        loss, logits = self.morph(hidden_states, labels)
        
        if not return_dict:
            return (loss,logits) + bert_outputs[2:]
        
        return MorphTaggingOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions,
        )

    def predict(self, sentences: List[str], tokenizer: BertTokenizerFast, padding='longest'):
        # tokenize the inputs and convert them to relevant device
        inputs = tokenizer(sentences, padding=padding, truncation=True, return_tensors='pt')
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        # calculate the logits
        logits = self.forward(**inputs, return_dict=True).logits
        return parse_logits(inputs['input_ids'].tolist(), sentences, tokenizer, logits)
    
def parse_logits(input_ids: List[List[int]], sentences: List[str], tokenizer: BertTokenizerFast, logits: MorphLogitsOutput):
    prefix_logits, pos_logits, feats_logits, suffix_logits, suffix_feats_logits = \
                logits.prefix_logits, logits.pos_logits, logits.features_logits, logits.suffix_logits, logits.suffix_features_logits

    prefix_predictions = (prefix_logits > 0.5).int().tolist() # Threshold at 0.5 for multi-label classification
    pos_predictions = pos_logits.argmax(axis=-1).tolist()
    suffix_predictions = suffix_logits.argmax(axis=-1).tolist()
    feats_predictions = [logits.argmax(axis=-1).tolist() for logits in feats_logits]
    suffix_feats_predictions = [logits.argmax(axis=-1).tolist() for logits in suffix_feats_logits]

    # create the return dictionary 
    # for each sentence, return a dict object with the following files { text, tokens }
    # Where tokens is a list of dicts, where each dict is: 
    #       { pos: str, feats: dict, prefixes: List[str], suffix: str | bool, suffix_feats: dict | None}
    special_toks = tokenizer.all_special_tokens
    special_toks.remove(tokenizer.unk_token)
    special_toks.remove(tokenizer.mask_token)
    
    ret = []
    for sent_idx,sentence in enumerate(sentences):
        input_id_strs = tokenizer.convert_ids_to_tokens(input_ids[sent_idx])
        # iterate through each token in the sentence, ignoring special tokens
        tokens = []
        for token_idx,token_str in enumerate(input_id_strs):
            if token_str in special_toks: continue
            if token_str.startswith('##'):
                tokens[-1]['token'] += token_str[2:]
                continue
            tokens.append(dict(
                token=token_str,
                pos=ALL_POS[pos_predictions[sent_idx][token_idx]],
                feats=get_features_dict_from_predictions(feats_predictions, (sent_idx, token_idx)),
                prefixes=[ALL_PREFIX_POS[idx] for idx,i in enumerate(prefix_predictions[sent_idx][token_idx]) if i > 0],
                suffix=get_suffix_or_false(ALL_SUFFIX_POS[suffix_predictions[sent_idx][token_idx]]),
            ))
            if tokens[-1]['suffix']:
                tokens[-1]['suffix_feats'] = get_features_dict_from_predictions(suffix_feats_predictions, (sent_idx, token_idx))
        ret.append(dict(text=sentence, tokens=tokens))
    return ret
    
def get_suffix_or_false(suffix):
    return False if suffix == 'none' else suffix

def get_features_dict_from_predictions(predictions, idx):
    ret = {}
    for (feat_idx, (feat_name, feat_values)) in enumerate(ALL_FEATURES):
        val = feat_values[predictions[feat_idx][idx[0]][idx[1]]]
        if val != 'none':
            ret[feat_name] = val
    return ret


