import torch
import numpy as np


def encode_data(dataset, tokenizer, max_seq_length=128):
    """Featurizes the dataset into input IDs and attention masks for input into a
     transformer-style model.

     NOTE: This method should featurize the entire dataset simultaneously,
     rather than row-by-row.

  Args:
    dataset: A Pandas dataframe containing the data to be encoded.
    tokenizer: A transformers.PreTrainedTokenizerFast object that is used to
      tokenize the data.
    max_seq_length: Maximum sequence length to either pad or truncate every
      input example to.

  Returns:
    input_ids: A PyTorch.Tensor (with dimensions [len(dataset), max_seq_length])
      containing token IDs for the data.
    attention_mask: A PyTorch.Tensor (with dimensions [len(dataset), max_seq_length])
      containing attention masks for the data.
  """
    ## TODO: Tokenize the questions and passages using both truncation and padding.
    ## Use the tokenizer provided in the argument and see the code comments above for
    ## more details.
    
    tokenized_examples = tokenizer(
        dataset["question"].tolist(),
        dataset["passage"].tolist(),
        truncation="only_second", # only truncate the context
        max_length=max_seq_length,
        padding="max_length",
        return_offsets_mapping=True,
    )
    
    input_ids, attention_mask = tokenized_examples['input_ids'], tokenized_examples['attention_mask']

    # we need tensors as a final result, so transfer to tensor
    input_ids = torch.tensor(np.array(input_ids), dtype=torch.long)
    attention_mask = torch.tensor(np.array(attention_mask), dtype=torch.long)
    
    # return results
    return input_ids, attention_mask


def extract_labels(dataset):
    """Converts labels into numerical labels.

  Args:
    dataset: A Pandas dataframe containing the labels in the column 'label'.

  Returns:
    labels: A list of integers corresponding to the labels for each example,
      where 0 is False and 1 is True.
  """
    ## TODO: Convert the labels to a numeric format and return as a list.
    
    # Initialize result list
    label_list = []
    
    # iterate every row and check if label for given index is True or False
    # update True -> 1, False -> 0
    for index in dataset['idx']:
        if dataset['label'][index] == True: 
            label_list.append(1)
        elif dataset['label'][index] == False:
            label_list.append(0)     

    return label_list
