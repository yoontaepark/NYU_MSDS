"""
TODO: Your code below!

This file should implement all steps described in Part 2, and can be structured however you want.

Rather than using normal BERT, you should use distilbert-base-uncased. This will train faster.

We recommend training on a GPU, either by using HPC or running the command line commands on Colab.

Hints:
    * It will probably be helpful to save intermediate outputs (preprocessed data).
    * To save your finetuned models, you can use torch.save().
"""
# importing libraries needed 
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from datasets import load_dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# referenced from below code link, it creates a Vocab class 
# https://github.com/clulab/releases/blob/e9ee2d41dc1b5992f8b0d6d748aff82de11f8495/lrec2020-pat/vocabulary.py#L111    
class Vocabulary:
    """Keeps a mapping between words and ids. Also keeps word counts."""
    
    def __init__(self):
        self.pad = 0
        self.unk = 1
        self.i2w = ['<pad>', '<unk>']
        self.w2i = {w:i for i,w in enumerate(self.i2w)}
        self.counts = [0] * len(self.i2w)

    def __len__(self):
        return len(self.i2w)

    def __iter__(self):
        return iter(self.i2w)

    def __contains__(self, word):
        if isinstance(word, int):
            return 0 <= word < len(self.i2w)
        else:
            return word in self.w2i

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in self.w2i:
                return self.index(key)
            else:
                return self.unk
        elif isinstance(key, (int, slice)):
            return self.i2w[key]
        elif key is None:
            return self.unk
        else:
            raise KeyError(f'invalid key: {type(key)} {key} {self.w2i} {self.i2w}')

    def prune(self, threshold):
        """returns a new vocabulary populated with elements
        from `self` whose count is greater than `threshold`"""
        v = Vocabulary()
        for i, w in enumerate(self.i2w):
            c = self.count(i)
            if c > threshold:
                v.count_word(w, c)
        return v

    def count_words(self, words):
        for w in words:
            self.count_word(w)

    def count_word(self, word, count=1):
        """increments the given word's count"""
        if word not in self.w2i:
            self.w2i[word] = len(self.i2w)
            self.i2w.append(word)
            self.counts.append(count)
        else:
            self.counts[self.w2i[word]] += count

    def word(self, i):
        """gets an index and returns a word"""
        return self.i2w[i]

    def index(self, w):
        """gets a word and returns an index"""
        return self.w2i.get(w, self.unk)

    def count(self, i):
        """gets an index (or word) and returns the corresponding count"""
        if isinstance(i, str):
            i = self.index(i)
        return self.counts[i] if 0 <= i < len(self.counts) else 0

    def histogram(self):
        """returns a list of (word, count) tuples"""
        return list(zip(self.i2w, self.counts))

    def __str__(self):
        return str(self.pad) + "\t" + str(self.unk) + "\t" + str(self.i2w) + "\t" + str(self.w2i) + "\t" + str(self.counts) + "\n"    

# revised this function. We just use sentence to create vocab for both deprel and rel_pos 
# https://github.com/clulab/releases/blob/e9ee2d41dc1b5992f8b0d6d748aff82de11f8495/lrec2020-pat/vocabulary.py#L111
def make_vocabulary(sentences, prune=0):
    """make dependency relation vocabulary from a list of sentences"""
    vocab = Vocabulary()
    for sentence in sentences:
        vocab.count_words(sentence)
    if prune > 0:
        vocab = vocab.prune(prune)
    return vocab

# created both functions to generate index of deprel and rel_pos 
def get_index_from_deprel(dataset, vocab):
    index_of_input = [vocab.index(x) for x in dataset['deprel']]
    return index_of_input

def get_index_from_relpos(dataset, vocab):
    index_of_input = [vocab.index(x) for x in dataset['rel_pos']]
    return index_of_input

# this function is to pad dep and rel pos 
# https://python.plainenglish.io/understanding-collate-fn-in-pytorch-f9d1742647d3
def custom_collate(data): 
    
    # while just taking each text in data, 
    text = [d['text'] for d in data] 
    
    # take rel pos index, and dep label index but add the fake label as 0 for cls, and sep token 
    rel = [torch.tensor([0] + d['rel_pos_idx'] + [0]) for d in data]
    dep = [torch.tensor([0] + d['dep_label_idx'] + [0]) for d in data]

    # then pad rel and dep to the longest length. use 0 as a padding index  
    rel = pad_sequence(rel, batch_first=True, padding_value=rel_pos_vocab.index('<pad>')) 
    dep = pad_sequence(dep, batch_first=True, padding_value=dep_vocab.index('<pad>')) 

    # return text, rel_pos, and dep 
    return { 
        'text': text,
        'rel_pos_idx': rel,
        'dep_label_idx': dep
    }

# define a finetuned classifier
# this function was referenced from lab 10 code
class FinetuneNliClassifier(nn.Module):

    # define final layers for both rel_pos_vocab(121), and dep(50)
    def __init__(self, model, d_hidden, n_class_1, n_class_2):
        super().__init__()
        self.model = model
        self.project1 = nn.Linear(d_hidden, n_class_1)
        self.project2 = nn.Linear(d_hidden, n_class_2)

    # add final layer after we pass hidden state
    def forward(self, sentence):
        hid_states , = self.model(**sentence).values() # [batch_size, seq_len, d_hidden]
        return self.project1(hid_states), self.project2(hid_states)   # [batch_size, seq_len, n_class_1], [batch_size, seq_len, n_class_2] 

# this function finetunes(trains) the dataset 
# batch size of 32 and learning rate of 1e-4
# this function was referenced from lab 10 code
def finetune_epoch(clf: FinetuneNliClassifier,
                   train_dataset,
                   lambda1,
                   batch_size=32,
                   lr=1e-4,
                   device="cuda:0"):
    
    # define a DataLoader batch size of 32. Note that we need to use custom padding function
    loader = DataLoader(train_dataset, batch_size, collate_fn=custom_collate)
  
    # set the model to train mode
    clf.train()

    # set parameters to unfreeze mode
    params_to_update = []
    for name,param in clf.named_parameters():
        if param.requires_grad == True: # this makes parameters unfreezed
            params_to_update.append(param)
            
    # set optimizer, and loss function (set ignore_index = 0, so that we can ignore padding, and also some other index that are assigned as 0)        
    optimizer = torch.optim.Adam(params_to_update, lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)               

    # set idx to 0
    idx = 0

    # for each batch, iterate and update the weight 
    for batch in tqdm(loader, desc="Train"):
        
        # initialize optimizer and loss 
        optimizer.zero_grad()
        loss = 0
        
        # encode text(sentence). This generates input_ids and attention_mask
        encoded_input = tokenizer(batch["text"], return_tensors="pt", padding=True).to(device)
        
        ## creating a fake labels based on the DistilBert tokenizer
        # for rel_pos and dep_rel, create a list 
        added_rel_pos_list = []
        added_dep_rel_list = []
        
        # for each list of labels, add fake labels as 0
        for i in range(len(batch['dep_label_idx'])):
            
            # step 1: tokenize the sentence by using DistilBert tokenizer
            tokenized_text = tokenizer.tokenize(batch['text'][i])
               
            # step 2: add cls and sep token manually 
            tokenized_text = ['[CLS]'] + tokenized_text + ['[SEP]']
            
            # step 3: filter out indexes that are not actually a token from golden label. We will be using these to create a fake label
            # initialize filtered index list 
            filtered_tokenized_index = []
            
            # for each index, check as below 
            for idx in range(len(tokenized_text)):
                
                # if idx == 0, it means that it is '[CLS]', so pass 
                # '[CLS]' is a token that needs to be indexed as fake label, but is already added when we call a collate function 
                if idx == 0:
                    pass
                
                # elif idx == 1, it means that it is the first word of the sentence
                # so need to check if it starts with '##' or, it is '-'
                elif idx == 1:
                    if tokenized_text[idx].startswith('##') or tokenized_text[idx] == '-':
                        filtered_tokenized_index.append(idx)
                
                # for other indexes, check if it starts with '##' or it is '-' or previous word is '-'
                else: 
                    if tokenized_text[idx].startswith('##') or tokenized_text[idx] == '-' or tokenized_text[idx-1] == '-':
                        filtered_tokenized_index.append(idx)
                        
            # add fake labels for rel pos
            # when the index of the rel pos matches with filtered tokenized index, append 0 and its value 
            # else, just append its value 
            added_rel_pos = []
            for j in range(batch['rel_pos_idx'][i].size()[0]):
                if j in filtered_tokenized_index:
                    added_rel_pos.append(0)
                    added_rel_pos.append(batch['rel_pos_idx'][i][j].item())
                else: 
                    added_rel_pos.append(batch['rel_pos_idx'][i][j].item())
            
            # after you updates rel pos label for one sentence, append to the initialized list (which was setup before)
            added_rel_pos_list.append(torch.tensor(added_rel_pos))
            
            # repeat for deprel
            added_dep_rel = []
            for j in range(batch['dep_label_idx'][i].size()[0]):
                if j in filtered_tokenized_index:
                    added_dep_rel.append(0)
                    added_dep_rel.append(batch['dep_label_idx'][i][j].item())
                else: 
                    added_dep_rel.append(batch['dep_label_idx'][i][j].item())
            added_dep_rel_list.append(torch.tensor(added_dep_rel))
        
        # padding is required once again, as the number of fake labels vary from labels 
        added_rel_pos_list = pad_sequence(added_rel_pos_list, batch_first=True, padding_value=rel_pos_vocab.index('<pad>')) 
        added_dep_rel_list = pad_sequence(added_dep_rel_list, batch_first=True, padding_value=dep_vocab.index('<pad>'))       
        
        ## now we are done with adding fake labels, run the model to get logits 
        logits_1, logits_2 = clf(encoded_input)
        
        # this is to make sure that we convert our labels into same device (eithe cpu or gpu)
        added_rel_pos_list = added_rel_pos_list.to(device)
        added_dep_rel_list = added_dep_rel_list.to(device)
        
        # assing min length for logjts and both labels 
        min_length = min(logits_1.size()[1], added_rel_pos_list.size()[1], added_dep_rel_list.size()[1])
        
        # filter both logits and labels, and change the logit's dimension as logit's second dimension has to be same as label's second dimension
        # see document: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_1 = criterion(logits_1[:, :min_length, :].permute(0,2,1), added_rel_pos_list[:, :min_length])
        loss_2 = criterion(logits_2[:, :min_length, :].permute(0,2,1), added_dep_rel_list[:, :min_length])
        
        # get weighted average loss, depending on the labmda
        loss += (lambda1 * loss_1) + ((1-lambda1) * loss_2)

        # backward the loss 
        loss.backward()
        
        # clip the gradient norm to maximum norm 1.0.
        nn.utils.clip_grad_norm_(clf.parameters(), max_norm=1.0, norm_type=2)

        # Perform a single optimization step (parameter update)
        optimizer.step()

        # for 10 idx, print loss to make sure if the model is training 
        if idx % 10 == 0:
            print(loss.item())
        idx += 1

# evaluate function, make sure this function doesn't update the weights 
# this function was referenced from lab 10 code
@torch.no_grad()
def evaluate(clf: FinetuneNliClassifier, eval_dataset, lambda1):
    
    # set the model to eval mode 
    clf.eval()

    # define a DataLoader batch size of 32. applying same logic to validation set 
    loader = DataLoader(eval_dataset, batch_size=32, collate_fn=custom_collate)

    # initialize correct and total counts for both rel_pos and dep rel 
    n_right_1 = 0
    n_total_1 = 0
    n_right_2 = 0
    n_total_2 = 0

    # for each batch, iterate and update the accuracy   
    for batch in tqdm(loader, desc="Eval"):
        
        # encode text(sentence). This generates input_ids and attention_mask
        encoded_input = tokenizer(batch["text"], return_tensors="pt", padding=True).to(device)
        
        ## creating a fake labels based on the DistilBert tokenizer
        # for rel_pos and dep_rel, create a list 
        added_rel_pos_list = []
        added_dep_rel_list = []        
        
        # for each list of labels, add fake labels as 0
        for i in range(len(batch['dep_label_idx'])):
            
            # step 1: tokenize the sentence by using DistilBert tokenizer            
            tokenized_text = tokenizer.tokenize(batch['text'][i])
            
            # step 2: add cls and sep token manually             
            tokenized_text = ['[CLS]'] + tokenized_text + ['[SEP]']
            
            # step 3: filter out indexes that are not actually a token from golden label. We will be using these to create a fake label
            # same logic as training set 
            filtered_tokenized_index = []
            for idx in range(len(tokenized_text)):
                if idx == 0:
                    pass
                elif idx == 1:
                    if tokenized_text[idx].startswith('##') or tokenized_text[idx] == '-':
                        filtered_tokenized_index.append(idx)                
                else: 
                    if tokenized_text[idx].startswith('##') or tokenized_text[idx] == '-' or tokenized_text[idx-1] == '-':
                        filtered_tokenized_index.append(idx)
                        
            # add fake labels for rel pos
            added_rel_pos = []
            for j in range(batch['rel_pos_idx'][i].size()[0]):
                if j in filtered_tokenized_index:
                    added_rel_pos.append(0)
                    added_rel_pos.append(batch['rel_pos_idx'][i][j].item())
                else: 
                    added_rel_pos.append(batch['rel_pos_idx'][i][j].item())
            
            added_rel_pos_list.append(torch.tensor(added_rel_pos))
            
            # repeat for deprel
            added_dep_rel = []
            for j in range(batch['dep_label_idx'][i].size()[0]):
                if j in filtered_tokenized_index:
                    added_dep_rel.append(0)
                    added_dep_rel.append(batch['dep_label_idx'][i][j].item())
                else: 
                    added_dep_rel.append(batch['dep_label_idx'][i][j].item())
            
            added_dep_rel_list.append(torch.tensor(added_dep_rel))
        
        # pad for rel_pos and dep_rel 
        added_rel_pos_list = pad_sequence(added_rel_pos_list, batch_first=True, padding_value=rel_pos_vocab.index('<pad>')) #(4)
        added_dep_rel_list = pad_sequence(added_dep_rel_list, batch_first=True, padding_value=dep_vocab.index('<pad>')) #(4)
        
        ## now we are done with adding fake labels, run the model to get logits 
        logits_1, logits_2 = clf(encoded_input)
        
        # this is to make sure that we convert our labels into same device (eithe cpu or gpu)
        added_rel_pos_list = added_rel_pos_list.to(device)
        added_dep_rel_list = added_dep_rel_list.to(device)
        
        # assign min length for logjts and both labels
        min_length = min(logits_1.size()[1], added_rel_pos_list.size()[1], added_dep_rel_list.size()[1])
        
        # filter both logits and labels, and argmax to get the prediction 
        # flatten both prediction and labels to compare 
        y_pred_1 = logits_1[:, :min_length, :].argmax(-1).flatten()
        y_actual_1 = added_rel_pos_list[:, :min_length].flatten()
        
        # for each (pred, actual) pair, only evaluate if actual is not zero 
        # if each pair is same, assign as true and add 'n_right' 
        # regardless of true or false, add 'n_total'
        for idx in range(y_actual_1.size()[0]):
            if y_actual_1[idx].item() != 0:
                right_1 = y_pred_1[idx] == y_actual_1[idx]
                n_right_1 += right_1.sum().item()
                n_total_1 += right_1.numel() 
        
        # apply this to dep rel 
        y_pred_2 = logits_2[:, :min_length, :].argmax(-1).flatten()
        y_actual_2 = added_dep_rel_list[:, :min_length].flatten()
        
        for idx in range(y_actual_2.size()[0]):
            if y_actual_2[idx].item() != 0:
                right_2 = y_pred_2[idx] == y_actual_2[idx]
                n_right_2 += right_2.sum().item()
                n_total_2 += right_2.numel()         
    
    # after all, calculate overall accuracy                 
    acc_1 = n_right_1 / n_total_1
    acc_2 = n_right_2 / n_total_2
                
    print("lambda: {}, Relative position Acc: {}, Dependency label Acc: {}".format(lambda1, acc_1, acc_2))
      

# finetune function: this is the main function that runs training and eval function by given epochs
# this function was referenced from lab 10 code 
def finetune(clf: FinetuneNliClassifier, dataset, lambda1, n_epochs: int = 3, **args):
    print("Using device:", args["device"])
    train = dataset["train"]
    valid = dataset["validation"]
    for epoch in range(n_epochs):
        print(f"Starting epoch {epoch}...")
        train = train.shuffle()
        finetune_epoch(clf, train, lambda1, **args)
        evaluate(clf, valid, lambda1)
    
    # after running the model, save the model 
    torch.save(clf, f'bert-parser-kogsd-{lambda1}.pt')
    
if __name__ == "__main__":

    # load dataset
    # dataset = load_dataset("universal_dependencies", "en_gum")   
    # below code is used when applying ko_gsd 
    dataset = load_dataset("universal_dependencies", "ko_gsd")   
    
    # for each head, assign rel_pos. rel_pos = head - token + 1 from the paper: https://aclanthology.org/2020.lrec-1.643.pdf
    list_of_rel_pos = []
    for data in dataset['train']:
        list_of_rel_pos.append([str(int(element2) - i + 1) if int(element2) != 0 else str(0) for i, element2 in enumerate(data['head'])])
    
    # add rel_pos column 
    dataset['train'] = dataset['train'].add_column(name='rel_pos', column = list_of_rel_pos)

    # load distilbert tokenizer and model 
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
   
    # this is to generate idx for each dep_label and rel_pos 
    dep_vocab = make_vocabulary(dataset['train']['deprel'])
    rel_pos_vocab = make_vocabulary(dataset['train']['rel_pos'])
    # print('dep_vocab: ', dep_vocab)
    # print('rel_pos_vocab: ', rel_pos_vocab)
    
    # add rel_pos_idx, and dep_label_index as a ground truth 
    dataset['train'] = dataset['train'].map(lambda x: {'rel_pos_idx': get_index_from_relpos(x, vocab=rel_pos_vocab)})
    dataset['train'] = dataset['train'].map(lambda x: {'dep_label_idx': get_index_from_deprel(x, vocab=dep_vocab)})
    
    # # this is only to save as tsv and print out the result 
    # train_df = pd.DataFrame(data = zip(dataset['train']['text'], dataset['train']['rel_pos'], dataset['train']['deprel']), columns=['text', 'rel_pos', 'dep_label'])
    # train_df.to_csv('en_gum_10.tsv', sep='\t', index=False, header=False)    
    
    # remove columns that would not be used
    remove_col = ['idx', 'tokens', 'lemmas', 'upos', 'xpos', 'feats', 'head', 'deps', 'misc', 'deprel', 'rel_pos']
    dataset['train'] = dataset['train'].remove_columns(remove_col)

    ## apply same preprocessing for validation set
    list_of_rel_pos = []
    for data in dataset['validation']:
        list_of_rel_pos.append([str(int(element2) - i + 1) if int(element2) != 0 else str(0) for i, element2 in enumerate(data['head'])])
    dataset['validation'] = dataset['validation'].add_column(name='rel_pos', column = list_of_rel_pos)
    
    dataset['validation'] = dataset['validation'].map(lambda x: {'rel_pos_idx': get_index_from_relpos(x, vocab=rel_pos_vocab)})
    dataset['validation'] = dataset['validation'].map(lambda x: {'dep_label_idx': get_index_from_deprel(x, vocab=dep_vocab)})
    
    remove_col = ['idx', 'tokens', 'lemmas', 'upos', 'xpos', 'feats', 'head', 'deps', 'misc', 'deprel', 'rel_pos']
    dataset['validation'] = dataset['validation'].remove_columns(remove_col)

    ## train and evaluate 
    # first, assign a model that has both rel_pos_vocab length (=121) and dep_vocab length (=50) as number of classes 
    classifier = FinetuneNliClassifier(model, 768, len(rel_pos_vocab), len(dep_vocab)) 
    
    # assign device and apply to the model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    classifier.to(device)
    
    # now run the model based on each labmda (0.25, 0.5, 0.75)
    finetune(classifier, dataset, lambda1=0.25, device=device)
    finetune(classifier, dataset, lambda1=0.5, device=device)
    finetune(classifier, dataset, lambda1=0.75, device=device)