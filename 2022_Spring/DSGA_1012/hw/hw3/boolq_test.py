import pandas as pd
import torch
import unittest

from boolq import BoolQDataset
from transformers import RobertaTokenizerFast

class TestBoolQDataset(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.dataset = pd.DataFrame.from_dict(
            {
                "question": ["question 0", "question 1"],
                "passage": ["passage 0", "passage 1"],
                "idx": [0, 1],
                "label": [True, False],
            }
        )
        self.max_seq_len = 4
        self.boolq_dataset = BoolQDataset(
            self.dataset, self.tokenizer, self.max_seq_len
        )

    def test_len(self):
        ## TODO: Test that the length of self.boolq_dataset is correct.
        ## len(self.boolq_dataset) should equal len(self.dataset).
        self.assertEqual(len(self.boolq_dataset), len(self.dataset))


    def test_item(self):
        ## TODO: Test that, for each element of self.boolq_dataset, 
        ## the output of __getitem__ (accessible via self.boolq_dataset[idx])
        ## has the correct keys, value dimensions, and value types.
        ## Each item should have keys ["input_ids", "attention_mask", "labels"].
        ## The input_ids and attention_mask values should both have length self.max_seq_len
        ## and type torch.long. The labels value should be a single numeric value.
        
        for i in range(len(self.boolq_dataset)):
            
            # the output of __getitem__ (accessible via self.boolq_dataset[idx]) 
            dict_ith = self.boolq_dataset[i]
            
            # checking keys
            self.assertEqual(list(dict_ith.keys()), ["input_ids", "attention_mask", "labels"])
            
            # The input_ids and attention_mask values should both have length self.max_seq_len
            self.assertEqual(list(dict_ith.values())[0].shape[0], self.max_seq_len)
            self.assertEqual(list(dict_ith.values())[1].shape[0], self.max_seq_len)            
            
            # and type torch.long. 
            self.assertEqual(list(dict_ith.values())[0].dtype, torch.long)
            self.assertEqual(list(dict_ith.values())[1].dtype, torch.long)
                
            # The labels value should be a single numeric value.
            self.assertEqual(type(list(dict_ith.values())[2]), int)
