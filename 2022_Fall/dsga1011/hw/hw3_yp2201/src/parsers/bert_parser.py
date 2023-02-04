from src.parsers.parser import Parser
from src.dependency_parse import DependencyParse
# from src.bert_parser_model import BertParserModel
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
import torch
import torch.nn as nn

import networkx as nx
from networkx.algorithms.tree.branchings import maximum_spanning_arborescence
from finetune_bert import FinetuneNliClassifier

class BertParser(Parser):

    """Represents a full parser that decodes parse trees from a finetuned BERT model."""

    def __init__(self, model_path: str, mst: bool = False):
        """Load your saved finetuned model using torch.load().

        Arguments:
            model_path: Path from which to load saved model.
            mst: Whether to use MST decoding or argmax decoding.
        """
        self.mst = mst
        # TODO: Load your neural net.
        
        # check gpu status and select either gpu/cpu based on the status 
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0: current_device = 'cuda' 
        else: current_device = 'cpu'       
                    
        # load entire model 
        self.model = torch.load(model_path, map_location=current_device)
        
        # set the model to eval mode 
        self.model.eval()
        
        # set current device, dep_vocab list, and rel_pos list
        # note that dep_vocab, and rel_pos_vocab lists are copied from part 2
        # below is for en_gum 
        self.device = current_device
        self.rel_pos_vocab = {'<pad>': 0, '<unk>': 1, '3': 2, '0': 3, '4': 4, '-1': 5, '-2': 6, '1': 7, '-6': 8, '-5': 9, '5': 10, '-3': 11, '8': 12, '-7': 13, '-4': 14, '-9': 15, '-15': 16, '17': 17, '14': 18, '7': 19, '6': 20, '-10': 21, '-13': 22, '10': 23, '9': 24, '18': 25, '-19': 26, '-11': 27, '-16': 28, '-8': 29, '-18': 30, '-45': 31, '-20': 32, '-14': 33, '-28': 34, '-30': 35, '-12': 36, '-59': 37, '11': 38, '16': 39, '13': 40, '-25': 41, '-26': 42, '32': 43, '30': 44, '26': 45, '-32': 46, '-51': 47, '-79': 48, '-56': 49, '-24': 50, '-47': 51, '-21': 52, '-17': 53, '21': 54, '20': 55, '-22': 56, '-40': 57, '12': 58, '-33': 59, '-36': 60, '15': 61, '-23': 62, '-58': 63, '-27': 64, '-82': 65, '19': 66, '-37': 67, '24': 68, '-41': 69, '-34': 70, '-39': 71, '22': 72, '-80': 73, '-81': 74, '-35': 75, '-29': 76, '-53': 77, '43': 78, '41': 79, '-43': 80, '37': 81, '34': 82, '-44': 83, '-31': 84, '23': 85, '-38': 86, '27': 87, '-72': 88, '-60': 89, '-55': 90, '-42': 91, '-71': 92, '47': 93, '-69': 94, '33': 95, '35': 96, '42': 97, '-63': 98, '-66': 99, '-46': 100, '28': 101, '31': 102, '-54': 103, '-49': 104, '-68': 105, '-70': 106, '-48': 107, '-88': 108, '25': 109, '-52': 110, '-65': 111, '38': 112, '-50': 113, '29': 114, '-61': 115, '-62': 116, '-85': 117, '-92': 118, '39': 119, '-57': 120}
        self.dep_vocab = {'<pad>': 0, '<unk>': 1, 'amod': 2, 'root': 3, 'cc': 4, 'conj': 5, 'punct': 6, 'case': 7, 'nmod': 8, 'flat': 9, 'list': 10, 'compound': 11, 'advmod': 12, 'aux': 13, 'nsubj': 14, 'obl': 15, 'det': 16, 'obj': 17, 'acl:relcl': 18, 'cop': 19, 'acl': 20, 'mark': 21, 'nmod:poss': 22, 'advcl': 23, 'nsubj:pass': 24, 'aux:pass': 25, 'compound:prt': 26, 'xcomp': 27, 'appos': 28, 'fixed': 29, 'parataxis': 30, 'csubj': 31, 'dep': 32, 'nmod:tmod': 33, 'nummod': 34, 'expl': 35, 'ccomp': 36, 'cc:preconj': 37, 'det:predet': 38, 'iobj': 39, 'orphan': 40, 'obl:tmod': 41, 'goeswith': 42, 'obl:npmod': 43, 'csubj:pass': 44, 'nmod:npmod': 45, 'vocative': 46, 'discourse': 47, 'dislocated': 48, 'reparandum': 49}
        
        # below is for ko_gsd
        # self.rel_pos_vocab = {'<pad>': 0, '<unk>': 1, '9': 2, '8': 3, '4': 4, '3': 5, '17': 6, '6': 7, '-5': 8, '1': 9, '7': 10, '0': 11, '11': 12, '5': 13, '-2': 14, '14': 15, '12': 16, '10': 17, '-6': 18, '-4': 19, '16': 20, '40': 21, '-3': 22, '20': 23, '13': 24, '-1': 25, '15': 26, '18': 27, '-7': 28, '-10': 29, '-16': 30, '19': 31, '23': 32, '21': 33, '27': 34, '25': 35, '28': 36, '26': 37, '-12': 38, '-8': 39, '-20': 40, '-13': 41, '33': 42, '-9': 43, '-11': 44, '22': 45, '29': 46, '35': 47, '34': 48, '62': 49, '-14': 50, '42': 51, '36': 52, '32': 53, '24': 54, '38': 55, '31': 56, '30': 57, '39': 58, '-15': 59, '-26': 60, '43': 61, '-19': 62, '48': 63, '-25': 64, '-18': 65, '49': 66, '37': 67, '58': 68, '-33': 69, '-17': 70, '-28': 71, '-41': 72, '-21': 73, '54': 74, '-24': 75, '-32': 76, '-37': 77, '-42': 78, '-46': 79, '-23': 80, '44': 81, '-27': 82, '41': 83, '-35': 84, '-22': 85, '46': 86, '51': 87, '-34': 88, '47': 89, '83': 90, '82': 91, '45': 92, '-38': 93, '55': 94, '-29': 95, '53': 96, '-30': 97, '-36': 98, '50': 99}
        # self.dep_vocab = {'<pad>': 0, '<unk>': 1, 'nsubj': 2, 'iobj': 3, 'obj': 4, 'advmod': 5, 'dep': 6, 'obl': 7, 'advcl': 8, 'det:poss': 9, 'conj': 10, 'punct': 11, 'flat': 12, 'acl:relcl': 13, 'root': 14, 'nmod': 15, 'appos': 16, 'case': 17, 'nummod': 18, 'nsubj:pass': 19, 'det': 20, 'amod': 21, 'ccomp': 22, 'cop': 23, 'mark': 24, 'cc': 25, 'fixed': 26, 'csubj': 27, 'aux': 28, 'acl': 29}
        
    def parse(self, sentence: str, tokens: list) -> DependencyParse:
        """Build a DependencyParse from the output of your loaded finetuned model.
        If self.mst == True, apply MST decoding. Otherwise use argmax decoding.        
        """
        # TODO: Your code here!

        # assign text and tokens as same as spacy parser
        res_dict = {
            "text": sentence,
            "tokens": tokens, 
            "head": [],
            "deprel": [],
        }
        
        # tokenize once agian, by using DistillBertTokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        # encode text(sentence). This generates input_ids and attention_mask
        encoded_input = tokenizer(res_dict['text'], return_tensors="pt", padding=True).to(self.device)
        
        # filter out indexes that are not actually a token from the golden truth
        # one thing that is different is that we are not adding cls and sep, as we can't make fake labels (labels are not available)
        tokenized_text = tokenizer.tokenize(res_dict['text'])
        
        # therefore, filtered index starts from index 0 to see if the word starts from '##' or is '-'
        # and from index 1, check to see if the word starts from '##' or is '-' or previous token is '-'
        filtered_tokenized_index = []
        for idx in range(len(tokenized_text)):

            if idx == 0:
                if tokenized_text[idx].startswith('##') or tokenized_text[idx] == '-':
                    filtered_tokenized_index.append(idx)
                                
            else: 
                if tokenized_text[idx].startswith('##') or tokenized_text[idx] == '-' or tokenized_text[idx-1] == '-':
                    filtered_tokenized_index.append(idx)        
        
        # run the model and get the logits
        logits_1, logits_2 = self.model(encoded_input)
        
        #### CASE1) argmax
        if not self.mst:
            # rel_pos: argmax and remove first and last result (which is CLS and SEP)
            rel_pos = logits_1.argmax(-1)
            rel_pos = rel_pos[:, 1:-1]
            
            # dep_rel: do same as rel_pos 
            dep_rel = logits_2.argmax(-1)
            dep_rel = dep_rel[:, 1:-1]
        
            # decode based on pre-defined vocab
            for i in range(dep_rel.size()[1]):
                
                # dep_rel: simply look at the value and find/append the key
                decoded_pred_dep = [x for x in self.dep_vocab if self.dep_vocab[x] == dep_rel[0][i]]
                res_dict['deprel'].append(decoded_pred_dep[0])

                # rel_pos: do same as dep_rel, and this will return the key value of rel_pos           
                decoded_pred_rel_pos = [x for x in self.rel_pos_vocab if self.rel_pos_vocab[x] == rel_pos[0][i]]
                
                # then, assign head from rel_pos
                # if dep_rel is 'root', then no need to calcuate so assign 0
                # else, as we calculated, get head from rel_pos, which is head = rel_pos + token - 1
                if decoded_pred_dep[0] == 'root':
                    res_dict['head'].append(str(0))
                else:
                    res_dict['head'].append(str(int(decoded_pred_rel_pos[0]) + i - 1))                 
            
        #### CASE2) mst  
        else:
            # dep_rel: applies same for both argmax and MST
            dep_rel = logits_2.argmax(-1)
            dep_rel = dep_rel[:, 1:-1]    
            
            # rel_pos
            # initialize graph 
            G = nx.DiGraph()
            
            # take log softmax for the logit 
            m = nn.LogSoftmax(dim=-1)
            logits_1 = m(logits_1)[0]
            
            # initialize rel_pos as zeros
            decoded_pred_rel_pos = ['0'] * logits_1.size()[0]

            # for number of tokens 
            for i in range(logits_1.size()[0]):
                
                # add node 
                G.add_node(i)
                
                # again, for number of tokens, add edges
                # this means that we are comparing (i, j) pairs
                for j in range(logits_1.size()[0]):
                    
                    # if i == j, then we assign no edges
                    # but for consistency, assign weight as 0.0 (this means no edge)
                    if i == j:
                        weight = 0.0
                    
                    # if i != j, we are assuming i as a head of j 
                    # that is, from j, i-j is the relative position 
                    # therefore, we firstly try to find the value in rel_pos_vocab. If no value exists, assign as value 1, which is unk 
                    else:
                        if str(i-j) in self.rel_pos_vocab.keys():
                            j_idx = self.rel_pos_vocab[str(i-j)]
                        else: 
                            j_idx = 1 ## unknown
                            
                        # assign weight as a logit that has a index of [j, j_idx], where j_idx is a relative position value in rel_pos_vocab    
                        weight = logits_1[j, j_idx].item()                    
                    
                    # add edge by assigned weight
                    G.add_edge(i, j, weight=weight)
            
            # run MST
            MST = maximum_spanning_arborescence(G)
            
            # for each MST edge pair (i, j)
            for edge in MST.edges:
                
                # update rel_pos by looking at j index (edge[1]) as str(i-j)
                decoded_pred_rel_pos[edge[1]] = str(edge[0] - edge[1])
                
            # remove cls and sep 
            decoded_pred_rel_pos = decoded_pred_rel_pos[1:-1]

            # decode based on pre-defined vocab
            for i in range(dep_rel.size()[1]):
                
                # dep_rel: same logic as argmax 
                decoded_pred_dep = [x for x in self.dep_vocab if self.dep_vocab[x] == dep_rel[0][i]]
                res_dict['deprel'].append(decoded_pred_dep[0])
                
                # assign head from rel_pos. Note that from MST, we've already decoded rel_pos 
                # if dep_rel is 'root', then no need to calcuate so assign 0
                # else, as we calculated, get head from rel_pos, which is head = rel_pos + token - 1
                if decoded_pred_dep[0] == 'root':
                    res_dict['head'].append(str(0))
                else:
                    res_dict['head'].append(str(int(decoded_pred_rel_pos[i]) + i - 1))                        
        
         
        ## for here, we instead filter prediction. This is due to the fact that we cannot truncate ground truth
        ## For training, it makes sense to create a fake ground truth labels and filter out altogether
        ## however, in this case, we don't know the answer so it cannot be work in that way
        # remove fake index results 
        deprel_filtered = []
        for i in range(len(res_dict['deprel'])):
            if i not in filtered_tokenized_index:
                deprel_filtered.append(res_dict['deprel'][i])
        
        head_filtered = []
        for i in range(len(res_dict['head'])):
            if i not in filtered_tokenized_index:
                head_filtered.append(res_dict['head'][i])
        
        # final adjustment
        # case1) len(pred) >= len(actual) : truncate prediction 
        # case2) len(pred) < len(actual) : append padding to prediction (needs to be always wrong, so assign '<pad>') 
        if len(deprel_filtered) > len(res_dict['tokens']):
            deprel_filtered = deprel_filtered[:len(res_dict['tokens'])]
            head_filtered = head_filtered[:len(res_dict['tokens'])]
        else:
            deprel_filtered += ['<pad>'] * (len(res_dict['tokens']) - len(deprel_filtered))
            head_filtered += ['<pad>'] * (len(res_dict['tokens']) - len(head_filtered))
        
        # after final adjustment, assign deprel and head 
        res_dict['deprel'] = deprel_filtered
        res_dict['head'] = head_filtered
        
        # return dependency parser 
        return DependencyParse.from_huggingface_dict(res_dict)
        
