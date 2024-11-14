from dataclasses import dataclass
import re
from operator import itemgetter
import torch
from torch import nn
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.utils import ModelOutput
from .BertForSyntaxParsing import BertSyntaxParsingHead, SyntaxLabels, SyntaxLogitsOutput, parse_logits as syntax_parse_logits
from .BertForPrefixMarking import BertPrefixMarkingHead, parse_logits as prefix_parse_logits, encode_sentences_for_bert_for_prefix_marking, get_prefixes_from_str
from .BertForMorphTagging import BertMorphTaggingHead, MorphLogitsOutput, MorphLabels, parse_logits as morph_parse_logits 
    
import warnings

@dataclass
class JointParsingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    # logits will contain the optional predictions for the given labels
    logits: Optional[Union[SyntaxLogitsOutput, None]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # if no labels are given, we will always include the syntax logits separately
    syntax_logits: Optional[SyntaxLogitsOutput] = None
    ner_logits: Optional[torch.FloatTensor] = None
    prefix_logits: Optional[torch.FloatTensor] = None
    lex_logits: Optional[torch.FloatTensor] = None
    morph_logits: Optional[MorphLogitsOutput] = None

# wrapper class to wrap a torch.nn.Module so that you can store a module in multiple linked
# properties without registering the parameter multiple times
class ModuleRef:
    def __init__(self, module: torch.nn.Module):
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

class BertForJointParsing(BertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config, do_syntax=None, do_ner=None, do_prefix=None, do_lex=None, do_morph=None, syntax_head_size=64):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # create all the heads as None, and then populate them as defined
        self.syntax, self.ner, self.prefix, self.lex, self.morph = (None,)*5

        if do_syntax is not None: 
            config.do_syntax = do_syntax
            config.syntax_head_size = syntax_head_size
        if do_ner is not None: config.do_ner = do_ner
        if do_prefix is not None: config.do_prefix = do_prefix
        if do_lex is not None: config.do_lex = do_lex
        if do_morph is not None: config.do_morph = do_morph
        
        # add all the individual heads
        if config.do_syntax:
            self.syntax = BertSyntaxParsingHead(config)
        if config.do_ner:
            self.num_labels = config.num_labels
            self.classifier = nn.Linear(config.hidden_size, config.num_labels) # name it same as in BertForTokenClassification 
            self.ner = ModuleRef(self.classifier)
        if config.do_prefix:
            self.prefix = BertPrefixMarkingHead(config)
        if config.do_lex:
            self.cls = BertOnlyMLMHead(config) # name it the same as in BertForMaskedLM
            self.lex = ModuleRef(self.cls)
        if config.do_morph:
            self.morph = BertMorphTaggingHead(config)
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder if self.lex is not None else None

    def set_output_embeddings(self, new_embeddings):
        if self.lex is not None:

            self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        prefix_class_id_options: Optional[torch.Tensor] = None,
        labels: Optional[Union[SyntaxLabels, MorphLabels, torch.Tensor]] = None,
        labels_type: Optional[Literal['syntax', 'ner', 'prefix', 'lex', 'morph']] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        compute_syntax_mst: Optional[bool] = None
    ):
        if return_dict is False:
            warnings.warn("Specified `return_dict=False` but the flag is ignored and treated as always True in this model.")
        
        if labels is not None and labels_type is None:
            raise ValueError("Cannot specify labels without labels_type")
        
        if labels_type == 'seg' and prefix_class_id_options is None:
            raise ValueError('Cannot calculate prefix logits without prefix_class_id_options')
        
        if compute_syntax_mst is not None and self.syntax is None:
            raise ValueError("Cannot compute syntax MST when the syntax head isn't loaded")


        bert_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        # calculate the extended attention mask for any child that might need it
        extended_attention_mask = None
        if attention_mask is not None:
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.size())

        # extract the hidden states, and apply the dropout
        hidden_states = self.dropout(bert_outputs[0])

        logits = None    
        syntax_logits = None
        ner_logits = None
        prefix_logits = None
        lex_logits = None
        morph_logits = None

        # Calculate the syntax
        if self.syntax is not None and (labels is None or labels_type == 'syntax'):
            # apply the syntax head
            loss, syntax_logits = self.syntax(hidden_states, extended_attention_mask, labels, compute_syntax_mst)
            logits = syntax_logits

        # Calculate the NER
        if self.ner is not None and (labels is None or labels_type == 'ner'):
            ner_logits = self.ner(hidden_states)
            logits = ner_logits
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # Calculate the segmentation
        if self.prefix is not None and (labels is None or labels_type == 'prefix'):
            loss, prefix_logits = self.prefix(hidden_states, prefix_class_id_options, labels)
            logits = prefix_logits
        
        # Calculate the lexeme
        if self.lex is not None and (labels is None or labels_type == 'lex'):
            lex_logits = self.lex(hidden_states)
            logits = lex_logits
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
                loss = loss_fct(lex_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if self.morph is not None and (labels is None or labels_type == 'morph'):
            loss, morph_logits = self.morph(hidden_states, labels)
            logits = morph_logits

        # no labels => logits = None
        if labels is None: logits = None

        return JointParsingOutput(
            loss,
            logits,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions,
            # all the predicted logits section
            syntax_logits=syntax_logits,
            ner_logits=ner_logits,
            prefix_logits=prefix_logits,
            lex_logits=lex_logits,
            morph_logits=morph_logits
        )

    def predict(self, sentences: Union[str, List[str]], tokenizer: BertTokenizerFast, padding='longest', truncation=True, compute_syntax_mst=True, per_token_ner=False, output_style: Literal['json', 'ud', 'iahlt_ud'] = 'json'):
        is_single_sentence = isinstance(sentences, str)
        if is_single_sentence:
            sentences = [sentences]
        
        if output_style not in ['json', 'ud', 'iahlt_ud']:
            raise ValueError('output_style must be in json/ud/iahlt_ud')
        if output_style in ['ud', 'iahlt_ud'] and (self.prefix is None or self.morph is None or self.syntax is None or self.lex is None):
            raise ValueError("Cannot output UD format when any of the prefix,morph,syntax, and lex heads aren't loaded.")
        
        # predict the logits for the sentence
        if self.prefix is not None:
            inputs = encode_sentences_for_bert_for_prefix_marking(tokenizer, self.config.prefix_cfg, sentences, padding)
        else:
            inputs = tokenizer(sentences, padding=padding, truncation=truncation, return_offsets_mapping=True, return_tensors='pt')

        offset_mapping = inputs.pop('offset_mapping')
        # Copy the tensors to the right device, and parse!
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        output = self.forward(**inputs, return_dict=True, compute_syntax_mst=compute_syntax_mst)
        
        input_ids = inputs['input_ids'].tolist() # convert once
        final_output = [dict(text=sentence, tokens=combine_token_wordpieces(ids, offsets, tokenizer)) for sentence, ids, offsets in zip(sentences, input_ids, offset_mapping)]
        # Syntax logits: each sentence gets a dict(tree: List[dict(word,dep_head,dep_head_idx,dep_func)], root_idx: int)
        if output.syntax_logits is not None:
            for sent_idx,parsed in enumerate(syntax_parse_logits(input_ids, sentences, tokenizer, output.syntax_logits)):
                merge_token_list(final_output[sent_idx]['tokens'], parsed['tree'], 'syntax')
                final_output[sent_idx]['root_idx'] = parsed['root_idx']
                
        # Prefix logits: each sentence gets a list([prefix_segment, word_without_prefix]) - **WITH CLS & SEP**
        if output.prefix_logits is not None:
            for sent_idx,parsed in enumerate(prefix_parse_logits(input_ids, sentences, tokenizer, output.prefix_logits, self.config.prefix_cfg)):
                merge_token_list(final_output[sent_idx]['tokens'], map(tuple, parsed[1:-1]), 'seg')
            
        # Lex logits each sentence gets a list(tuple(word, lexeme))
        if output.lex_logits is not None:
            for sent_idx, parsed in enumerate(lex_parse_logits(input_ids, sentences, tokenizer, output.lex_logits)):
                merge_token_list(final_output[sent_idx]['tokens'], map(itemgetter(1), parsed), 'lex')
                
        # morph logits each sentences get a dict(text=str, tokens=list(dict(token, pos, feats, prefixes, suffix, suffix_feats?)))
        if output.morph_logits is not None:
            for sent_idx,parsed in enumerate(morph_parse_logits(input_ids, sentences, tokenizer, output.morph_logits)):
                merge_token_list(final_output[sent_idx]['tokens'], parsed['tokens'], 'morph')
            
        # NER logits each sentence gets a list(tuple(word, ner))
        if output.ner_logits is not None:
            for sent_idx,parsed in enumerate(ner_parse_logits(input_ids, sentences, tokenizer, output.ner_logits, self.config.id2label)):
                if per_token_ner:
                    merge_token_list(final_output[sent_idx]['tokens'], map(itemgetter(1), parsed), 'ner')
                final_output[sent_idx]['ner_entities'] = aggregate_ner_tokens(final_output[sent_idx], parsed) 
        
        if output_style in ['ud', 'iahlt_ud']:
            final_output = convert_output_to_ud(final_output, self.config, style='htb' if output_style == 'ud' else 'iahlt')
        
        if is_single_sentence:
            final_output = final_output[0]
        return final_output



def aggregate_ner_tokens(final_output, parsed):
    entities = []
    prev = None
    for token_idx, (d, (word, pred)) in enumerate(zip(final_output['tokens'], parsed)):
        # O does nothing
        if pred == 'O': prev = None
        # B- || I-entity != prev (different entity or none)
        elif pred.startswith('B-') or pred[2:] != prev:
            prev = pred[2:]
            entities.append([[word], dict(label=prev, start=d['offsets']['start'], end=d['offsets']['end'], token_start=token_idx, token_end=token_idx)])
        else: 
            entities[-1][0].append(word)
            entities[-1][1]['end'] = d['offsets']['end']
            entities[-1][1]['token_end'] = token_idx

    return [dict(phrase=' '.join(words), **d) for words, d in entities]

def merge_token_list(src, update, key):
    for token_src, token_update in zip(src, update):
        token_src[key] = token_update
        
def combine_token_wordpieces(input_ids: List[int], offset_mapping: torch.Tensor, tokenizer: BertTokenizerFast):
    offset_mapping = offset_mapping.tolist()
    ret = []
    special_toks = tokenizer.all_special_tokens
    special_toks.remove(tokenizer.unk_token)
    special_toks.remove(tokenizer.mask_token)
    
    for token, offsets in zip(tokenizer.convert_ids_to_tokens(input_ids), offset_mapping):
        if token in special_toks: continue
        if token.startswith('##'):
            ret[-1]['token'] += token[2:]
            ret[-1]['offsets']['end'] = offsets[1]
        else: ret.append(dict(token=token, offsets=dict(start=offsets[0], end=offsets[1])))
    return ret

def ner_parse_logits(input_ids: List[List[int]], sentences: List[str], tokenizer: BertTokenizerFast, logits: torch.Tensor, id2label: Dict[int, str]):
    predictions = torch.argmax(logits, dim=-1).tolist()
    batch_ret = []

    special_toks = tokenizer.all_special_tokens
    special_toks.remove(tokenizer.unk_token)
    special_toks.remove(tokenizer.mask_token)
    
    for batch_idx in range(len(sentences)):

        ret = []
        batch_ret.append(ret)

        tokens = tokenizer.convert_ids_to_tokens(input_ids[batch_idx])
        for tok_idx in range(len(tokens)):
            token = tokens[tok_idx]
            if token in special_toks: continue

            # wordpieces should just be appended to the previous word
            # we modify the last token in ret
            # by discarding the original end position and replacing it with the new token's end position
            if token.startswith('##'):
                continue
            # for each token, we append a tuple containing: token, label, start position, end position
            ret.append((token, id2label[predictions[batch_idx][tok_idx]]))

    return batch_ret

def lex_parse_logits(input_ids: List[List[int]], sentences: List[str], tokenizer: BertTokenizerFast, logits: torch.Tensor):
    
    predictions = torch.argsort(logits, dim=-1, descending=True)[..., :3].tolist()
    batch_ret = []

    special_toks = tokenizer.all_special_tokens
    special_toks.remove(tokenizer.unk_token)
    special_toks.remove(tokenizer.mask_token)
    for batch_idx in range(len(sentences)):
        intermediate_ret = []
        tokens = tokenizer.convert_ids_to_tokens(input_ids[batch_idx])
        for tok_idx in range(len(tokens)):
            token = tokens[tok_idx]
            if token in special_toks: continue

            # wordpieces should just be appended to the previous word
            if token.startswith('##'):
                intermediate_ret[-1] = (intermediate_ret[-1][0] + token[2:], intermediate_ret[-1][1])
                continue
            intermediate_ret.append((token, tokenizer.convert_ids_to_tokens(predictions[batch_idx][tok_idx])))
        
        # build the final output taking into account valid letters
        ret = []
        batch_ret.append(ret)
        for (token, lexemes) in intermediate_ret:
            # must overlap on at least 2 non אהוי letters
            possible_lets = set(c for c in token if c not in 'אהוי')
            final_lex = '[BLANK]'
            for lex in lexemes:
                if sum(c in possible_lets for c in lex) >= min([2, len(possible_lets), len([c for c in lex if c not in 'אהוי'])]): 
                    final_lex = lex
                    break
            ret.append((token, final_lex))
        
    return batch_ret

ud_prefixes_to_pos = {
    'ש': ['SCONJ'],
    'מש': ['SCONJ'],
    'כש': ['SCONJ'],
    'לכש': ['SCONJ'],
    'בש': ['SCONJ'],
    'לש': ['SCONJ'],
    'ו': ['CCONJ'],
    'ל': ['ADP'],
    'ה': ['DET', 'SCONJ'],
    'מ': ['ADP', 'SCONJ'],
    'ב': ['ADP'],
    'כ': ['ADP', 'ADV'],
}
ud_suffix_to_htb_str = {
    'Gender=Masc|Number=Sing|Person=3': '_הוא',	
	'Gender=Masc|Number=Plur|Person=3': '_הם',	
	'Gender=Fem|Number=Sing|Person=3': '_היא',	
	'Gender=Fem|Number=Plur|Person=3': '_הן',	
	'Gender=Fem,Masc|Number=Plur|Person=1': '_אנחנו',	
	'Gender=Fem,Masc|Number=Sing|Person=1': '_אני',	
	'Gender=Masc|Number=Plur|Person=2': '_אתם',	
	'Gender=Masc|Number=Sing|Person=3': '_הוא',
	'Gender=Masc|Number=Sing|Person=2': '_אתה',	
	'Gender=Fem|Number=Sing|Person=2': '_את',	
	'Gender=Masc|Number=Plur|Person=3': '_הם'
}
def convert_output_to_ud(output_sentences, model_cfg, style: Literal['htb', 'iahlt']):
    if style not in ['htb', 'iahlt']:
        raise ValueError('style must be htb/iahlt')
        
    final_output = []
    for sent_idx, sentence in enumerate(output_sentences):
        # next, go through each word and insert it in the UD format. Store in a temp format for the post process
        intermediate_output = []
        ranges = []
        # store a mapping between each word index and the actual line it appears in
        idx_to_key = {-1: 0}
        for word_idx,word in enumerate(sentence['tokens']):
            try:
                # handle blank lexemes
                if word['lex'] == '[BLANK]':
                    word['lex'] = word['seg'][-1]
            except KeyError:
                import json
                print(json.dumps(sentence, ensure_ascii=False, indent=2))
                exit(0)

            start = len(intermediate_output)
            # Add in all the prefixes
            if len(word['seg']) > 1:
                for pre in get_prefixes_from_str(word['seg'][0], model_cfg.prefix_cfg, greedy=True):
                    # pos - just take the first valid pos that appears in the predicted prefixes list. 
                    pos = next((pos for pos in ud_prefixes_to_pos[pre] if pos in word['morph']['prefixes']), ud_prefixes_to_pos[pre][0])
                    dep, func = ud_get_prefix_dep(pre, word, word_idx)
                    intermediate_output.append(dict(word=pre, lex=pre, pos=pos, dep=dep, func=func, feats='_'))
                    
                # if there was an implicit heh, add it in dependent on the method
                if not 'ה' in pre and intermediate_output[-1]['pos'] == 'ADP' and 'DET' in word['morph']['prefixes']:
                    if style == 'htb':
                        intermediate_output.append(dict(word='ה_', lex='ה', pos='DET', dep=word_idx, func='det', feats='_'))
                    elif style == 'iahlt':
                        intermediate_output[-1]['feats'] = 'Definite=Def|PronType=Art'
                    
                
            idx_to_key[word_idx] = len(intermediate_output) + 1
            # add the main word in!
            intermediate_output.append(dict(
                    word=word['seg'][-1], lex=word['lex'], pos=word['morph']['pos'], 
                    dep=word['syntax']['dep_head_idx'], func=word['syntax']['dep_func'],
                    feats='|'.join(f'{k}={v}' for k,v in word['morph']['feats'].items())))
            
            # if we have suffixes, this changes things
            if word['morph']['suffix']:
                # first determine the dependency info:
                # For adp, num, det - they main word points to here, and the suffix points to the dependency
                entry_to_assign_suf_dep = None
                if word['morph']['pos'] in ['ADP', 'NUM', 'DET']:
                    entry_to_assign_suf_dep = intermediate_output[-1]
                    intermediate_output[-1]['func'] = 'case'
                    dep = word['syntax']['dep_head_idx']
                    func = word['syntax']['dep_func']
                else:                
                    # if pos is verb -> obj, num -> dep, default to -> nmod:poss
                    dep = word_idx
                    func = {'VERB': 'obj', 'NUM': 'dep'}.get(word['morph']['pos'], 'nmod:poss')
                
                s_word, s_lex = word['seg'][-1], word['lex']
                # update the word of the string and extract the string of the suffix!
                # for IAHLT:
                if style == 'iahlt':
                    # we need to shorten the main word and extract the suffix
                    # if it is longer than the lexeme - just take off the lexeme.
                    if len(s_word) > len(s_lex):
                        idx = len(s_lex)
                    # Otherwise, try to find the last letter of the lexeme, and fail that just take the last letter
                    else:
                        # take either len-1, or the last occurence (which can be -1 === len-1)
                        idx = min([len(s_word) - 1, s_word.rfind(s_lex[-1])])
                    # extract the suffix and update the main word
                    suf = s_word[idx:]
                    intermediate_output[-1]['word'] = s_word[:idx]
                # for htb:
                elif style == 'htb':
                    # main word becomes the lexeme, the suffix is based on the features
                    intermediate_output[-1]['word'] = (s_lex if s_lex != s_word else s_word[:-1]) + '_'
                    suf_feats = word['morph']['suffix_feats']
                    suf = ud_suffix_to_htb_str.get(f"Gender={suf_feats.get('Gender', 'Fem,Masc')}|Number={suf_feats.get('Number', 'Sing')}|Person={suf_feats.get('Person', '3')}", "_הוא")
                    # for HTB, if the function is poss, then add a shel pointing to the next word
                    if func == 'nmod:poss' and s_lex != 'של':
                        intermediate_output.append(dict(word='_של_', lex='של', pos='ADP', dep=len(intermediate_output) + 2, func='case', feats='_', absolute_dep=True))
                # add the main suffix in
                intermediate_output.append(dict(word=suf, lex='הוא', pos='PRON', dep=dep, func=func, feats='|'.join(f'{k}={v}' for k,v in word['morph']['suffix_feats'].items())))
                if entry_to_assign_suf_dep:
                    entry_to_assign_suf_dep['dep'] = len(intermediate_output)
                    entry_to_assign_suf_dep['absolute_dep'] = True
                    
            end = len(intermediate_output)
            ranges.append((start, end, word['token']))
            
        # now that we have the intermediate output, combine it to the final output
        cur_output = []
        final_output.append(cur_output)
        # first, add the headers 
        cur_output.append(f'# sent_id = {sent_idx + 1}')
        cur_output.append(f'# text = {sentence["text"]}')
        
        # add in all the actual entries
        for start,end,token in ranges:
            if end - start > 1:
                cur_output.append(f'{start + 1}-{end}\t{token}\t_\t_\t_\t_\t_\t_\t_\t_')
            for idx,output in enumerate(intermediate_output[start:end], start + 1):
                # compute the actual dependency location
                dep = output['dep'] if output.get('absolute_dep', False) else idx_to_key[output['dep']]
                func = normalize_dep_rel(output['func'], style)
                # and add the full ud string in
                cur_output.append('\t'.join([
                    str(idx),
                    output['word'],
                    output['lex'],
                    output['pos'], 
                    output['pos'],
                    output['feats'],
                    str(dep),
                    func,
                    '_', '_'
                ]))
    return final_output
                
def normalize_dep_rel(dep, style: Literal['htb', 'iahlt']):
    if style == 'iahlt':
        if dep == 'compound:smixut': return 'compound'
        if dep == 'nsubj:cop': return 'nsubj'
        if dep == 'mark:q': return 'mark'
        if dep == 'case:gen' or dep == 'case:acc': return 'case'
    return dep
    
    
def ud_get_prefix_dep(pre, word, word_idx):
    does_follow_main = False
    
    # shin goes to the main word for verbs, otherwise follows the word
    if pre.endswith('ש'): 
        does_follow_main = word['morph']['pos'] != 'VERB'
        func = 'mark'
    # vuv goes to the main word if the function is in the list, otherwise follows
    elif pre == 'ו': 
        does_follow_main = word['syntax']['dep_func'] not in ["conj", "acl:recl", "parataxis", "root", "acl", "amod", "list", "appos", "dep", "flatccomp"]
        func = 'cc'
    else:
        # for adj, noun, propn, pron, verb - prefixes go to the main word
        if word['morph']['pos'] in ["ADJ", "NOUN", "PROPN", "PRON", "VERB"]:
            does_follow_main = False
        # otherwise - prefix follows the word if the function is in the list
        else: does_follow_main = word['syntax']['dep_func'] in ["compound:affix", "det", "aux", "nummod", "advmod", "dep", "cop", "mark", "fixed"]
        
        func = 'case'
        if pre == 'ה':
            func = 'det' if 'DET' in word['morph']['prefixes'] else 'mark'

    return (word['syntax']['dep_head_idx'] if does_follow_main else word_idx), func