import math
from transformers.utils import ModelOutput
import torch
from torch import nn
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast

ALL_FUNCTION_LABELS = ["nsubj", "nsubj:cop", "punct", "mark", "mark:q", "case", "case:gen", "case:acc", "fixed", "obl", "det", "amod", "acl:relcl", "nmod", "cc", "conj", "root", "compound:smixut", "cop", "compound:affix", "advmod", "nummod", "appos", "nsubj:pass", "nmod:poss", "xcomp", "obj", "aux", "parataxis", "advcl", "ccomp", "csubj", "acl", "obl:tmod", "csubj:pass", "dep", "dislocated", "nmod:tmod", "nmod:npmod", "flat", "obl:npmod", "goeswith", "reparandum", "orphan", "list", "discourse", "iobj", "vocative", "expl", "flat:name"]

@dataclass
class SyntaxLogitsOutput(ModelOutput):
    dependency_logits: torch.FloatTensor = None
    function_logits: torch.FloatTensor = None
    dependency_head_indices: torch.LongTensor = None

    def detach(self):
        return SyntaxTaggingOutput(self.dependency_logits.detach(), self.function_logits.detach(), self.dependency_head_indices.detach())

@dataclass
class SyntaxTaggingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[SyntaxLogitsOutput] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class SyntaxLabels(ModelOutput):
    dependency_labels: Optional[torch.LongTensor] = None
    function_labels: Optional[torch.LongTensor] = None

    def detach(self):
        return SyntaxLabels(self.dependency_labels.detach(), self.function_labels.detach())
    
    def to(self, device):
        return SyntaxLabels(self.dependency_labels.to(device), self.function_labels.to(device))

class BertSyntaxParsingHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # the attention query & key values
        self.head_size = config.syntax_head_size# int(config.hidden_size / config.num_attention_heads * 2)
        self.query = nn.Linear(config.hidden_size, self.head_size)
        self.key = nn.Linear(config.hidden_size, self.head_size)
        # the function classifier gets two encoding values and predicts the labels
        self.num_function_classes = len(ALL_FUNCTION_LABELS)
        self.cls = nn.Linear(config.hidden_size * 2, self.num_function_classes)

    def forward(
            self, 
            hidden_states: torch.Tensor, 
            extended_attention_mask: Optional[torch.Tensor],
            labels: Optional[SyntaxLabels] = None,
            compute_mst: bool = False) -> Tuple[torch.Tensor, SyntaxLogitsOutput]:
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.head_size)

        # add in the attention mask
        if extended_attention_mask is not None:
            if extended_attention_mask.ndim == 4:
                extended_attention_mask = extended_attention_mask.squeeze(1)
            attention_scores += extended_attention_mask# batch x seq x seq

        # At this point take the hidden_state of the word and of the dependency word, and predict the function
        # If labels are provided, use the labels.
        if self.training and labels is not None:
            # Note that the labels can have -100, so just set those to zero with a max
            dep_indices = labels.dependency_labels.clamp_min(0)
        # Otherwise - check if he wants the MST or just the argmax
        elif compute_mst:
            dep_indices = compute_mst_tree(attention_scores, extended_attention_mask)
        else: 
            dep_indices = torch.argmax(attention_scores, dim=-1)
        
        # After we retrieved the dependency indicies, create a tensor of teh batch indices, and and retrieve the vectors of the heads to calculate the function
        batch_indices = torch.arange(dep_indices.size(0)).view(-1, 1).expand(-1, dep_indices.size(1)).to(dep_indices.device)
        dep_vectors = hidden_states[batch_indices, dep_indices, :] # batch x seq x dim

        # concatenate that with the last hidden states, and send to the classifier output
        cls_inputs = torch.cat((hidden_states, dep_vectors), dim=-1)
        function_logits = self.cls(cls_inputs)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # step 1: dependency scores loss - this is applied to the attention scores
            loss = loss_fct(attention_scores.view(-1, hidden_states.size(-2)), labels.dependency_labels.view(-1))
            # step 2: function loss
            loss += loss_fct(function_logits.view(-1, self.num_function_classes), labels.function_labels.view(-1))
        
        return (loss, SyntaxLogitsOutput(attention_scores, function_logits, dep_indices))


class BertForSyntaxParsing(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.syntax = BertSyntaxParsingHead(config)
        
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[SyntaxLabels] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        compute_syntax_mst: Optional[bool] = None,
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
        
        extended_attention_mask = None
        if attention_mask is not None:
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.size())
        # apply the syntax head
        loss, logits = self.syntax(self.dropout(bert_outputs[0]), extended_attention_mask, labels, compute_syntax_mst)
        
        if not return_dict:
            return (loss,(logits.dependency_logits, logits.function_logits)) + bert_outputs[2:]
        
        return SyntaxTaggingOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions,
        )

    def predict(self, sentences: Union[str, List[str]], tokenizer: BertTokenizerFast, compute_mst=True):
        if isinstance(sentences, str):
            sentences = [sentences]
            
        # predict the logits for the sentence
        inputs = tokenizer(sentences, padding='longest', truncation=True, return_tensors='pt')
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        logits = self.forward(**inputs, return_dict=True, compute_syntax_mst=compute_mst).logits
        return parse_logits(inputs['input_ids'].tolist(), sentences, tokenizer, logits)

def parse_logits(input_ids: List[List[int]], sentences: List[str], tokenizer: BertTokenizerFast, logits: SyntaxLogitsOutput):
    outputs = []        

    special_toks = tokenizer.all_special_tokens
    special_toks.remove(tokenizer.unk_token)
    special_toks.remove(tokenizer.mask_token)
    
    for i in range(len(sentences)):
        deps = logits.dependency_head_indices[i].tolist()
        funcs = logits.function_logits.argmax(-1)[i].tolist()
        toks = [tok for tok in tokenizer.convert_ids_to_tokens(input_ids[i]) if tok not in special_toks]
        
        # first, go through the tokens and create a mapping between each dependency index and the index without wordpieces
        # wordpieces. At the same time, append the wordpieces in
        idx_mapping = {-1:-1} # default root 
        real_idx = -1
        for i in range(len(toks)):
            if not toks[i].startswith('##'):
                real_idx += 1
            idx_mapping[i] = real_idx
            
        # build our tree, keeping tracking of the root idx
        tree = []
        root_idx = 0
        for i in range(len(toks)):
            if toks[i].startswith('##'):
                tree[-1]['word'] += toks[i][2:]
                continue 
            
            dep_idx = deps[i + 1] - 1 # increase 1 for cls, decrease 1 for cls
            if dep_idx == len(toks): dep_idx = i - 1 # if he predicts sep, then just point to the previous word
            
            dep_head = 'root' if dep_idx == -1 else toks[dep_idx]
            dep_func = ALL_FUNCTION_LABELS[funcs[i + 1]]

            if dep_head == 'root': root_idx = len(tree)
            tree.append(dict(word=toks[i], dep_head_idx=idx_mapping[dep_idx], dep_func=dep_func))
        # append the head word
        for d in tree:
            d['dep_head'] = tree[d['dep_head_idx']]['word']
        
        outputs.append(dict(tree=tree, root_idx=root_idx))
    return outputs


def compute_mst_tree(attention_scores: torch.Tensor, extended_attention_mask: torch.LongTensor):
    # attention scores should be 3 dimensions - batch x seq x seq (if it is 2 - just unsqueeze)
    if attention_scores.ndim == 2: attention_scores = attention_scores.unsqueeze(0)
    if attention_scores.ndim != 3 or attention_scores.shape[1] != attention_scores.shape[2]:
        raise ValueError(f'Expected attention scores to be of shape batch x seq x seq, instead got {attention_scores.shape}')
    
    batch_size, seq_len, _ = attention_scores.shape
    # start by softmaxing so the scores are comparable
    attention_scores = attention_scores.softmax(dim=-1)
    
    batch_indices = torch.arange(batch_size, device=attention_scores.device)
    seq_indices = torch.arange(seq_len, device=attention_scores.device)

    seq_lens = torch.full((batch_size,), seq_len)
    
    if extended_attention_mask is not None:
        seq_lens = torch.argmax((extended_attention_mask != 0).int(), dim=2).squeeze(1)     
        # zero out any padding
        attention_scores[extended_attention_mask.squeeze(1) != 0] = 0

    # set the values for the CLS and sep to all by very low, so they never get chosen as a replacement arc
    attention_scores[:, 0, :] = 0
    attention_scores[batch_indices, seq_lens - 1, :] = 0
    attention_scores[batch_indices, :, seq_lens - 1] = 0 # can never predict sep
    # set the values for each token pointing to itself be 0
    attention_scores[:, seq_indices, seq_indices] = 0
    
    # find the root, and make him super high so we never have a conflict
    root_cands = torch.argsort(attention_scores[:, :, 0], dim=-1)
    attention_scores[batch_indices.unsqueeze(1), root_cands, 0] = 0
    attention_scores[batch_indices, root_cands[:, -1], 0] = 1.0

    # we start by getting the argmax for each score, and then computing the cycles and contracting them
    sorted_indices = torch.argsort(attention_scores, dim=-1, descending=True)
    indices = sorted_indices[:, :, 0].clone() # take the argmax
    
    attention_scores = attention_scores.tolist()
    seq_lens = seq_lens.tolist()
    sorted_indices = [[sub_l[:slen] for sub_l in l[:slen]] for l,slen in zip(sorted_indices.tolist(), seq_lens)]


    # go through each batch item and make sure our tree works
    for batch_idx in range(batch_size):
        # We have one root - detect the cycles and contract them. A cycle can never contain the root so really
        # for every cycle, we look at all the nodes, and find the highest arc out of the cycle for any values. Replace that and tada
        has_cycle, cycle_nodes = detect_cycle(indices[batch_idx], seq_lens[batch_idx])
        contracted_arcs = set()
        while has_cycle:
            base_idx, head_idx = choose_contracting_arc(indices[batch_idx], sorted_indices[batch_idx], cycle_nodes, contracted_arcs, seq_lens[batch_idx], attention_scores[batch_idx])
            indices[batch_idx, base_idx] = head_idx
            contracted_arcs.add(base_idx)
            # find the next cycle
            has_cycle, cycle_nodes = detect_cycle(indices[batch_idx], seq_lens[batch_idx])

    return indices

def detect_cycle(indices: torch.LongTensor, seq_len: int):
    # Simple cycle detection algorithm
    # Returns a boolean indicating if a cycle is detected and the nodes involved in the cycle
    visited = set()
    for node in range(1, seq_len - 1): # ignore the CLS/SEP tokens
        if node in visited:
            continue
        current_path = set()
        while node not in visited:
            visited.add(node)
            current_path.add(node)
            node = indices[node].item()
            if node == 0: break # roots never point to anything
            if node in current_path:
                return True, current_path  # Cycle detected
    return False, None

def choose_contracting_arc(indices: torch.LongTensor, sorted_indices: List[List[int]], cycle_nodes: set, contracted_arcs: set, seq_len: int, scores: List[List[float]]):
    # Chooses the highest-scoring, non-cycling arc from a graph. Iterates through 'cycle_nodes' to find
    # the best arc based on 'scores', avoiding cycles and zero node connections.
    # For each node, we only look at the next highest scoring non-cycling arc 
    best_base_idx, best_head_idx = -1, -1
    score = 0
    
    # convert the indices to a list once, to avoid multiple conversions (saves a few seconds)
    currents = indices.tolist()
    for base_node in cycle_nodes:
        if base_node in contracted_arcs: continue
        # we don't want to take anything that has a higher score than the current value - we can end up in an endless loop
        # Since the indices are sorted, as soon as we find our current item, we can move on to the next. 
        current = currents[base_node]
        found_current = False
        
        for head_node in sorted_indices[base_node]:
            if head_node == current:
                found_current = True
                continue
            if head_node in contracted_arcs: continue
            if not found_current or head_node in cycle_nodes or head_node == 0: 
                continue
            
            current_score = scores[base_node][head_node]
            if current_score > score:
                best_base_idx, best_head_idx, score = base_node, head_node, current_score
            break
    
    if best_base_idx == -1:
        raise ValueError('Stuck in endless loop trying to compute syntax mst. Please try again setting compute_syntax_mst=False')

    return best_base_idx, best_head_idx