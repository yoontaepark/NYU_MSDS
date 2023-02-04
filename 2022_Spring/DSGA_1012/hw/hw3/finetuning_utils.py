from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import RobertaForSequenceClassification

def compute_metrics(eval_pred):
    """Computes accuracy, f1, precision, and recall from a 
    transformers.trainer_utils.EvalPrediction object.
    """
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    ## TODO: Return a dictionary containing the accuracy, f1, precision, and recall scores.
    ## You may use sklearn's precision_recall_fscore_support and accuracy_score methods.
    
    # initialize a dictionary
    res_dict = {}
    
    # calculate accuracy, f1, precision, and recall scores
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    
    # add the result in pre-defined dictionary
    res_dict['accuracy'] = accuracy
    res_dict['f1'] = f1
    res_dict['precision'] = precision
    res_dict['recall'] = recall

    # return dictionary 
    return res_dict

def model_init():
    """Returns an initialized model for use in a Hugging Face Trainer."""
    ## TODO: Return a pretrained RoBERTa model for sequence classification.
    ## See https://huggingface.co/transformers/model_doc/roberta.html#robertaforsequenceclassification.
    
    model = RobertaForSequenceClassification.from_pretrained("roberta-base")
    
    return model
