from argparse import ArgumentParser
from collections import defaultdict

from src.dependency_parse import DependencyParse
from src.parsers.bert_parser import BertParser
from src.parsers.spacy_parser import SpacyParser
from src.metrics import get_metrics

# add libraries 
import numpy as np
from datasets import load_dataset
from typing import List

## note that this has to be imported to use saved model 
from finetune_bert import FinetuneNliClassifier

def get_parses(subset: str, test: bool = False) -> List[DependencyParse]:
    """Return a list of dependency parses in language specified by `subset` from the universal_dependencies dataset.py

    You should use HuggingFaceDatasets to load the dataset.
    
    Return the test set of test == True; validation set otherwise.
    """
    # TODO: Your code here!
    dataset = load_dataset("universal_dependencies", subset)
    
    # create an empty list 
    list_of_dep_parse = [] 
    
    # if test == True, return the test set        
    if test == True:
        # for each test dataset, convert to dependency parser class and append 
        for data in dataset['test']:
            list_of_dep_parse.append(DependencyParse.from_huggingface_dict(data))          
        return list_of_dep_parse
    
    # else, return the val set 
    else:
        # for each val dataset, convert to dependency parser class and append 
        for data in dataset['validation']:
            list_of_dep_parse.append(DependencyParse.from_huggingface_dict(data))          
        return list_of_dep_parse


def parse_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("method", choices=["spacy", "bert"])
    arg_parser.add_argument("--data_subset", type=str, default="zh_gsdsimp")
    arg_parser.add_argument("--test", action="store_true")

    # SpaCy parser arguments.
    arg_parser.add_argument("--model_name", type=str, default="zh_core_web_sm")

    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.method == "spacy":
        parser = SpacyParser(args.model_name)
    elif args.method == "bert":
        # replace model path, and boolean value to run different lambda with different decoding method
        # also, I've replaced with kogsd model for extra credit question 
        file_path = 'bert-parser-0.75.pt'
        # file_path = 'bert-parser-kogsd-0.75.pt'
        parser = BertParser(file_path, False)
        # parser = BertParser(file_path, True)
    else:
        raise ValueError("Unknown parser")
    
    cum_metrics = defaultdict(list)
    for gold in get_parses(args.data_subset, test=args.test):
        pred = parser.parse(gold.text, gold.tokens)
        for metric, value in get_metrics(pred, gold).items():
            cum_metrics[metric].append(value)
    print({metric: np.mean(data) for metric, data in cum_metrics.items()})
