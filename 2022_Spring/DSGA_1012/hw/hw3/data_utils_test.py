import data_utils
import pandas as pd
import torch
import unittest

from transformers import RobertaTokenizerFast


class TestDataUtils(unittest.TestCase):
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

    def test_sample(self):
        ## An example of a basic unit test, using class variables initialized in
        ## setUpClass().
        self.assertEqual(self.max_seq_len, 4)

    def test_encode_data(self):
        ## TODO: Write a unit test that asserts that the dimensions and dtype of the
        ## output of encode_data() are correct.
        ## input_ids should have shape [len(self.dataset), self.max_seq_len] and type torch.long.
        ## attention_mask should have the same shape and type.
        
        # recalling function
        input_ids, attention_mask = data_utils.encode_data(self.dataset, self.tokenizer, max_seq_length=self.max_seq_len)
        
        # checking dimensions: [len(self.dataset), self.max_seq_len] 
        # checking dtype: torch.long. == torch.int64, and checking by torch.int64        
        self.assertEqual(list(input_ids.shape), [self.dataset.shape[0], self.max_seq_len])
        self.assertEqual(input_ids.dtype, torch.long)
        
        # checking dimensions: [len(self.dataset), self.max_seq_len] 
        # checking dtype: torch.long. == torch.int64, and checking by torch.int64
        self.assertEqual(list(attention_mask.shape), [self.dataset.shape[0], self.max_seq_len])
        self.assertEqual(attention_mask.dtype, torch.long)

    def test_extract_labels(self):
        ## TODO: Write a unit test that asserts that extract_labels() outputs the
        ## correct labels, e.g. [1, 0].
      
        label_list = data_utils.extract_labels(self.dataset)
        
        # checking dimensions: [len(self.dataset), self.max_seq_len] 
        # checking dtype: torch.long. == torch.int64, and checking by torch.int64
        self.assertEqual(len(label_list), self.dataset.shape[0])
        for label_i in label_list:
            self.assertEqual(type(label_i), int)


if __name__ == "__main__":
    unittest.main()

