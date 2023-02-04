# Extra credit question
# Findings
'''
Just changing algoritm to BasicVariantGenerator, I could try random/grid search.


'''

import argparse
import boolq
import data_utils
import finetuning_utils
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast
from transformers import TrainingArguments, Trainer 

# Just adding Random search and grid search libraray
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray import tune

parser = argparse.ArgumentParser(
    description="Run a hyperparameter search for finetuning a RoBERTa model on the BoolQ dataset."
)
parser.add_argument(
    "data_dir",
    type=str,
    help="Directory containing the BoolQ dataset. Can be downloaded from https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip.",
)

args = parser.parse_args()

# Since the labels for the test set have not been released, we will use half of the
# validation dataset as our test dataset for the purposes of this assignment.
train_df = pd.read_json(f"{args.data_dir}/train.jsonl", lines=True, orient="records")
val_df, test_df = train_test_split(
    pd.read_json(f"{args.data_dir}/val.jsonl", lines=True, orient="records"),
    test_size=0.5,
)

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
train_data = boolq.BoolQDataset(train_df, tokenizer)
val_data = boolq.BoolQDataset(val_df, tokenizer)
test_data = boolq.BoolQDataset(test_df, tokenizer)

## TODO: Initialize a transformers.TrainingArguments object here for use in
## training and tuning the model. Consult the assignment handout for some
## sample hyperparameter values.

training_args = TrainingArguments(
    output_dir="./test",
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3, # due to time/computation constraints (should change to 3)
    logging_steps=500,
    logging_first_step=True,
    save_steps=1000,
    evaluation_strategy = "epoch", # evaluate at the end of every epoch
    learning_rate= 1e-5,
    weight_decay=0.01,
)

## TODO: Initialize a transformers.Trainer object and run a Bayesian
## hyperparameter search for at least 5 trials (but not too many) on the 
## learning rate. Hint: use the model_init() and
## compute_metrics() methods from finetuning_utils.py as arguments to
## Trainer(). Use the hp_space parameter in hyperparameter_search() to specify
## your hyperparameter search space. (Note that this parameter takes a function
## as its value.)
## Also print out the run ID, objective value,
## and hyperparameters of your best run.

# Setting Trainer object
trainer = Trainer(
    model_init = finetuning_utils.model_init,
    args = training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    compute_metrics=finetuning_utils.compute_metrics
)


# Setting hyperparameter 
def hp_config(arg):
    return {
    "learning_rate": tune.uniform(1e-5, 5e-5)}

# changing Random search and grid search as another hyperparameter search 
best_trial = trainer.hyperparameter_search(
    hp_space = hp_config, 
    compute_objective = lambda metrics : metrics["eval_accuracy"],
    direction="maximize",
    backend="ray",
    search_alg=BasicVariantGenerator(metric="eval_loss", mode="min"),
    n_trials = 5,
)

# prints out final result
print(best_trial)
