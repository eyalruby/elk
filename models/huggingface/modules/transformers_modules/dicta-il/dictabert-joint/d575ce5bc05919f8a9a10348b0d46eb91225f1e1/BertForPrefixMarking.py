from transformers.utils import ModelOutput
import torch
from torch import nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast

# define the classes, and the possible prefixes for each class
POSSIBLE_PREFIX_CLASSES =  [ ['לכש', 'כש', 'מש', 'בש', 'לש'], ['מ'], ['ש'], ['ה'], ['ו'], ['כ'], ['ל'], ['ב'] ]
POSSIBLE_RABBINIC_PREFIX_CLASSES =  [ ['לכש', 'כש', 'מש', 'בש', 'לש', 'לד', 'בד', 'מד', 'כד', 'לכד'], ['מ'], ['ש', 'ד'], ['ה'], ['ו'], ['כ'], ['ל'], ['ב'], ['א'], ['ק'] ]

class PrefixConfig(dict):
    def __init__(self, possible_classes, **kwargs): # added kwargs for previous version where all features were kept as dict values
        super().__init__()
        self.possible_classes = possible_classes
        self.total_classes = len(possible_classes)
        self.prefix_c2i = {w: i for i, l in enumerate(possible_classes) for w in l}
        self.all_prefix_items = list(sorted(self.prefix_c2i.keys(), key=len, reverse=True))
    
    @property
    def possible_classes(self) -> List[List[str]]:
        return self.get('possible_classes')
    
    @possible_classes.setter
    def possible_classes(self, value: List[List[str]]):
        self['possible_classes'] = value
        
DEFAULT_PREFIX_CONFIG = PrefixConfig(POSSIBLE_PREFIX_CLASSES)

def get_prefixes_from_str(s, cfg: PrefixConfig, greedy=False):
    # keep trimming prefixes from the string
    while len(s) > 0 and s[0] in cfg.prefix_c2i:
        # find the longest string to trim
        next_pre = next((pre for pre in cfg.all_prefix_items if s.startswith(pre)), None)
        if next_pre is None:
            return
        yield next_pre
        # if the chosen prefix is more than one letter, there is always an option that the 
        # prefix is actually just the first letter of the prefix - so offer that up as a valid prefix
        # as well. We will still jump to the length of the longer one, since if the next two/three
        # letters are a prefix, they have to be the longest one
        if not greedy and len(next_pre) > 1:
            yield next_pre[0]
        s = s[len(next_pre):]

def get_prefix_classes_from_str(s, cfg: PrefixConfig, greedy=False):
    for pre in get_prefixes_from_str(s, cfg, greedy):
        yield cfg.prefix_c2i[pre]

@dataclass
class PrefixesClassifiersOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class BertPrefixMarkingHead(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        if not hasattr(config, 'prefix_cfg') or config.prefix_cfg is None:
            setattr(config, 'prefix_cfg', DEFAULT_PREFIX_CONFIG)
        if isinstance(config.prefix_cfg, dict):
            config.prefix_cfg = PrefixConfig(config.prefix_cfg['possible_classes'])

        # an embedding table containing an embedding for each prefix class + 1 for NONE
        # we will concatenate either the embedding/NONE for each class - and we want the concatenate
        # size to be the hidden_size
        prefix_class_embed = config.hidden_size // config.prefix_cfg.total_classes
        self.prefix_class_embeddings = nn.Embedding(config.prefix_cfg.total_classes + 1, prefix_class_embed)
        
        # one layer for transformation, apply an activation, then another N classifiers for each prefix class
        self.transform = nn.Linear(config.hidden_size + prefix_class_embed * config.prefix_cfg.total_classes, config.hidden_size)
        self.activation = nn.Tanh()
        self.classifiers = nn.ModuleList([nn.Linear(config.hidden_size, 2) for _ in range(config.prefix_cfg.total_classes)])

    def forward(
            self,
            hidden_states: torch.Tensor,
            prefix_class_id_options: torch.Tensor,
            labels: Optional[torch.Tensor] = None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        
        # encode the prefix_class_id_options
        # If input_ids is batch x seq_len
        # Then sequence_output is batch x seq_len x hidden_dim
        # So prefix_class_id_options is batch x seq_len x total_classes
        # Looking up the embeddings should give us batch x seq_len x total_classes x hidden_dim / N
        possible_class_embed = self.prefix_class_embeddings(prefix_class_id_options)
        # then flatten the final dimension - now we have batch x seq_len x hidden_dim_2
        possible_class_embed = possible_class_embed.reshape(possible_class_embed.shape[:-2] + (-1,))

        # concatenate the new class embed into the sequence output before the transform
        pre_transform_output = torch.cat((hidden_states, possible_class_embed), dim=-1) # batch x seq_len x (hidden_dim + hidden_dim_2)
        pre_logits_output = self.activation(self.transform(pre_transform_output))# batch x seq_len x hidden_dim

        # run each of the classifiers on the transformed output
        logits = torch.cat([cls(pre_logits_output).unsqueeze(-2) for cls in self.classifiers], dim=-2)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        
        return (loss, logits)
        


class BertForPrefixMarking(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.prefix = BertPrefixMarkingHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        prefix_class_id_options: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
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

        loss, logits = self.prefix.forward(hidden_states, prefix_class_id_options, labels)
        if not return_dict:
            return (loss,logits,) + bert_outputs[2:]

        return PrefixesClassifiersOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions,
        )
    
    def predict(self, sentences: List[str], tokenizer: BertTokenizerFast, padding='longest'):
        # step 1: encode the sentences through using the tokenizer, and get the input tensors + prefix id tensors
        inputs = encode_sentences_for_bert_for_prefix_marking(tokenizer, self.config.prefix_cfg, sentences, padding)
        inputs.pop('offset_mapping')
        inputs = {k:v.to(self.device) for k,v in inputs.items()}

        # run through bert
        logits = self.forward(**inputs, return_dict=True).logits
        return parse_logits(inputs['input_ids'].tolist(), sentences, tokenizer, logits, self.config.prefix_cfg)

def parse_logits(input_ids: List[List[int]], sentences: List[str], tokenizer: BertTokenizerFast, logits: torch.FloatTensor, config: PrefixConfig):
    # extract the predictions by argmaxing the final dimension (batch x sequence x prefixes x prediction)
    logit_preds = torch.argmax(logits, axis=3).tolist()

    ret = []

    for sent_idx,sent_ids in enumerate(input_ids):
        tokens = tokenizer.convert_ids_to_tokens(sent_ids)
        ret.append([])
        for tok_idx,token in enumerate(tokens):
            # If we've reached the pad token, then we are at the end
            if token == tokenizer.pad_token: continue
            if token.startswith('##'): continue

            # combine the next tokens in? only if it's a breakup
            next_tok_idx = tok_idx + 1
            while next_tok_idx < len(tokens) and tokens[next_tok_idx].startswith('##'):
                token += tokens[next_tok_idx][2:]
                next_tok_idx += 1

            prefix_len = get_predicted_prefix_len_from_logits(token, logit_preds[sent_idx][tok_idx], config)
        
            if not prefix_len:
                ret[-1].append([token])
            else:
                ret[-1].append([token[:prefix_len], token[prefix_len:]])
    return ret

def encode_sentences_for_bert_for_prefix_marking(tokenizer: BertTokenizerFast, config: PrefixConfig, sentences: List[str], padding='longest', truncation=True):
    inputs = tokenizer(sentences, padding=padding, truncation=truncation, return_offsets_mapping=True, return_tensors='pt')
    # create our prefix_id_options array which will be like the input ids shape but with an addtional
    # dimension containing for each prefix whether it can be for that word
    prefix_id_options = torch.full(inputs['input_ids'].shape + (config.total_classes,), config.total_classes, dtype=torch.long)

    # go through each token, and fill in the vector accordingly
    for sent_idx, sent_ids in enumerate(inputs['input_ids']):
        tokens = tokenizer.convert_ids_to_tokens(sent_ids)
        for tok_idx, token in enumerate(tokens):
            # if the first letter isn't a valid prefix letter, nothing to talk about
            if len(token) < 2 or not token[0] in config.prefix_c2i: continue

            # combine the next tokens in? only if it's a breakup
            next_tok_idx = tok_idx + 1
            while next_tok_idx < len(tokens) and tokens[next_tok_idx].startswith('##'):
                token += tokens[next_tok_idx][2:]
                next_tok_idx += 1

            # find all the possible prefixes - and mark them as 0 (and in the possible mark it as it's value for embed lookup)
            for pre_class in get_prefix_classes_from_str(token, config):
                prefix_id_options[sent_idx, tok_idx, pre_class] = pre_class
        
    inputs['prefix_class_id_options'] = prefix_id_options
    return inputs

def get_predicted_prefix_len_from_logits(token, token_logits, config: PrefixConfig):
    # Go through each possible prefix, and check if the prefix is yes - and if
    # so increase the counter of the matched length, otherwise break out. That will solve cases
    # of predicting prefix combinations that don't exist on the word.
    # For example, if we have the word ושכשהלכתי and the model predict ו & כש, then we will only
    # take the vuv because in order to get the כש we need the ש as well.
    # Two extra items:
    # 1] Don't allow the same prefix multiple times
    # 2] Always check that the word starts with that prefix - otherwise it's bad 
    #    (except for the case of multi-letter prefix, where we force the next to be last)
    cur_len, skip_next, last_check, seen_prefixes = 0, False, False, set()
    for prefix in get_prefixes_from_str(token, config):
        # Are we skipping this prefix? This will be the case where we matched כש, don't allow ש
        if skip_next:
            skip_next = False
            continue
        # check for duplicate prefixes, we don't allow two of the same prefix
        # if it predicted two of the same, then we will break out
        if prefix in seen_prefixes: break
        seen_prefixes.add(prefix)

        # check if we predicted this prefix
        if token_logits[config.prefix_c2i[prefix]]:
            cur_len += len(prefix)
            if last_check: break
            skip_next = len(prefix) > 1
        # Otherwise, we predicted no. If we didn't, then this is the end of the prefix
        # and time to break out. *Except* if it's a multi letter prefix, then we allow
        # just the next letter - e.g., if כש doesn't match, then we allow כ, but then we know
        # the word continues with a ש, and if it's not כש, then it's not כ-ש- (invalid)
        elif len(prefix) > 1:
            last_check = True
        else:
            break

    return cur_len
