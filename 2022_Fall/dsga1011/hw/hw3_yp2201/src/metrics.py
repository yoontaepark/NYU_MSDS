from src.dependency_parse import DependencyParse
from collections import defaultdict

def get_metrics(predicted: DependencyParse, labeled: DependencyParse) -> dict:
    # TODO: Your code here!
    
    # initialize result dictionary 
    res_dict = defaultdict(list)
    
    # initialize uas_cnt, and las_cnt as zero
    uas_cnt = 0
    las_cnt = 0
    
    # iterate number of ground truths (labeled)
    for i in range(len(labeled.heads)):
        
        # if head is correct, add uas cnt 
        if predicted.heads[i] == labeled.heads[i]:
            uas_cnt += 1
            # if deprel is also correct, add las cnt 
            if predicted.deprel[i] == labeled.deprel[i]:
                las_cnt += 1
    
    # divide correct count by total count              
    uas_cnt_prop = uas_cnt / len(labeled.heads)
    las_cnt_prop = las_cnt / len(labeled.deprel)
    
    # add each prop to uas, las 
    res_dict['uas'] = uas_cnt_prop
    res_dict['las'] = las_cnt_prop

    # return updated dictionary 
    # print('SCORE', res_dict)
    return res_dict
