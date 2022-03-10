import data_utils
import torch

from torch.utils.data import Dataset


class BoolQDataset(Dataset):
    """
    A torch.utils.data.Dataset wrapper for the BoolQ dataset.
    """

    def __init__(self, dataframe, tokenizer, max_seq_length=256):
        """max_seq_length
        Args:
          dataframe: A Pandas dataframe containing the data.
          tokenizer: A transformers.PreTrainedTokenizerFast object that is used to
            tokenize the data.
          max_seq_length: Maximum sequence length to either pad or truncate every
            input example to.
        """
        ## TODO: Use encode_data() from data_utils to store the input IDs and 
        ## attention masks for the data.
        
        # Use encode_data() from data_utils to store the input IDs and attention masks
        self.input_ids, self.attention_mask = data_utils.encode_data(dataframe, tokenizer, \
                                                                     max_seq_length=max_seq_length)
        
        ## TODO: Use extract_labels() from data_utils to store the labels.
        self.label_list = data_utils.extract_labels(dataframe)

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, i):
        """
        Returns:
          example: A dictionary containing the input_ids, attention_mask, and
            label for the i-th example, with the values being numeric tensors
            and the keys being 'input_ids', 'attention_mask', and 'labels'.
        """
        ## TODO: Return the i-th example as a dictionary with the keys and values
        ## specified in the function docstring. You should be able to extract the
        ## necessary values from self.encoded_data and self.label_list.
        
        # define a dictionary
        dict_ith = {}
        
        # make a key, value pair of 'input_ids', 'attention_mask', and 'labels'
        dict_ith['input_ids'] = self.input_ids[i]
        dict_ith['attention_mask'] = self.attention_mask[i]
        dict_ith['labels'] = self.label_list[i]        

        return dict_ith 
