# README
** Note: Collaborated with yc5206, yk2678, and yl7143

## Part 1
Hand in: In README.md, report the UAS and LAS for your SpaCy parser on en_gum.

Ans) 
spacy parser w en_gum: 'uas': 0.571266862839058, 'las': 0.43277008078703616
(For reference, w zh_gsdsimp: 'uas': 0.5020046927332706, 'las': 0.18073711469458392)

## Part 2
1. Hand in: Print the first 10 preprocessed examples to the file en_gum_10.tsv in the format text\trel_pos\tdep_label, 
where \t represents a tab, text is the input text, and rel_pos and dep_label are the two different label lists, 
serialized by calling str() on the Python lists

Ans) 
Aesthetic Appreciation and Spanish Art:	['3', '0', '4', '3', '-1', '-2']	['amod', 'root', 'cc', 'amod', 'conj', 'punct']
Insights from Eye-Tracking	['0', '3', '0']	['root', 'case', 'nmod']
Claire Bailey-Ross claire.bailey-ross@port.ac.uk University of Portsmouth, United Kingdom	['0', '1', '0', '-1', '3', '0', '4', '3', '-6']	['root', 'flat', 'list', 'list', 'case', 'nmod', 'punct', 'amod', 'list']
Andrew Beresford a.m.beresford@durham.ac.uk Durham University, United Kingdom	['0', '1', '0', '3', '-2', '4', '3', '-5']	['root', 'flat', 'list', 'compound', 'list', 'punct', 'amod', 'list']
Daniel Smith daniel.smith2@durham.ac.uk Durham University, United Kingdom	['0', '1', '0', '3', '-2', '4', '3', '-5']	['root', 'flat', 'list', 'compound', 'list', 'punct', 'amod', 'list']
Claire Warwick c.l.h.warwick@durham.ac.uk Durham University, United Kingdom	['0', '1', '0', '3', '-2', '4', '3', '-5']	['root', 'flat', 'list', 'compound', 'list', 'punct', 'amod', 'list']
How do people look at and experience art?	['5', '4', '3', '0', '5', '3', '-1', '-2', '-3']	['advmod', 'aux', 'nsubj', 'root', 'case', 'cc', 'conj', 'obl', 'punct']
Which elements of specific artworks do they focus on?	['3', '8', '4', '3', '-1', '4', '3', '0', '-5', '0']	['det', 'obl', 'case', 'amod', 'nmod', 'aux', 'nsubj', 'root', 'case', 'punct']
Do museum labels have an impact on how people look at artworks?	['5', '3', '3', '0', '3', '0', '3', '0', '3', '0', '3', '0', '-7']	['aux', 'compound', 'nsubj', 'root', 'det', 'obj', 'case', 'nmod', 'nsubj', 'acl:relcl', 'case', 'obl', 'punct']
The viewing experience of art is a complex one, involving issues of perception, attention, memory, decision-making, affect, and emotion.	['4', '3', '8', '3', '0', '5', '4', '3', '0', '3', '0', '1', '3', '0', '3', '0', '3', '-2', '3', '-4', '3', '-6', '4', '3', '-9', '-15']	['det', 'compound', 'nsubj', 'case', 'nmod', 'cop', 'det', 'amod', 'root', 'punct', 'acl', 'obj', 'case', 'nmod', 'punct', 'conj', 'punct', 'conj', 'punct', 'conj', 'punct', 'conj', 'punct', 'cc', 'conj', 'punct']

2. Hand in: Make sure to save final checkpoints for each of the three models using torch.save(), with the filename bert-parser-λ.pt, where λ is replaced by one of {.25, .5, .75}. Also report the validation accuracy for predicting both relative head position and the dependency arc label, for each model (that is, 2 accuracy numbers for 3 different models).

Ans) 
Saved final checkpoints by name bert-parser-0.25.pt, bert-parser-0.50.pt, bert-parser-0.75.pt
Validation accuracy for both relative head position and the dependency arc label
- lambda: 0.25, Relative position Acc: 0.6488652391332222, Dependency label Acc: 0.8238235671239903
- lambda: 0.5, Relative position Acc: 0.7120784715989229, Dependency label Acc: 0.8350429542249006
- lambda: 0.75, Relative position Acc: 0.7327862546480318, Dependency label Acc: 0.8378638286959866


## Part 3
Hand in: UAS/LAS on validation set (x6 total), UAS/LAS on test set (x2 total)

Ans) 
For test set, I've selected lambda 0.75, as uas/las showed highest among three different lambdas
- val set, argmax, lambda: 0.25, 'uas': 0.5290119535455057, 'las': 0.5080685610873038
- val set, argmax, lambda: 0.5, 'uas': 0.5752640329013106, 'las': 0.5496637858627477
- val set, argmax, lambda: 0.75, 'uas': 0.585607161162004, 'las': 0.5599867710313531

- val set, MST, lambda: 0.25, 'uas': 0.4955412584232084, 'las': 0.47347491546190146
- val set, MST, lambda: 0.5, 'uas': 0.5359603231557685, 'las': 0.5113458630770067
- val set, MST, lambda: 0.75, 'uas': 0.5557852502025515, 'las': 0.5309912896137173

- test set, argmax, lambda: 0.75, 'uas': 0.5932568033707905, 'las': 0.5651240170308404
- test set, MST, lambda: 0.75, 'uas': 0.5646485590633155, 'las': 0.5384299958599996


## Extra credit: 
Describe the language you pick and report UAS/LAS on validation set (x6 total), UAS/LAS on test set (x2 total), discussion of results in comparison to English

Ans) 
I've selected ko_gsd as I was interested in Korean language processing 

For part1, I've ran below code, so using ko_core_news_sm model in spacy to evaluate ko_gsd dataset
python eval_parser.py spacy --data_subset ko_gsd --model_name ko_core_news_sm
spacy parser: 'uas': 0.528721023773134, 'las': 0.3116088761752858

Below are part2 performance results for reference 
lambda: 0.25, Relative position Acc: 0.3832580699113564, Dependency label Acc: 0.31050342866700115
lambda: 0.5, Relative position Acc: 0.4040809499916374, Dependency label Acc: 0.3528181970229135
lambda: 0.75, Relative position Acc: 0.4120254223114233, Dependency label Acc: 0.35833751463455427

Running part3, I've got below results. Based on 'las' score, I've selected lambda 0.75 for test set 
- val set, argmax, lambda: 0.25, 'uas': 0.09348221070107215, 'las': 0.05531768030006205
- val set, argmax, lambda: 0.5, 'uas': 0.09264757701005467, 'las': 0.05996670169815964
- val set, argmax, lambda: 0.75: 'uas': 0.08968714726111394, 'las': 0.06127145498146336

- val set, MST, lambda: 0.25, 'uas': 0.09329597588325839, 'las': 0.055131445482248284
- val set, MST, lambda: 0.5, 'uas': 0.09118750803419624, 'las': 0.059549547702058274
- val set, MST, lambda: 0.75, 'uas': 0.08775366090249088, 'las': 0.06044204102103036

- test set, argmax, lambda: 0.75, 'uas': 0.08928309722252276, 'las': 0.06026576640732955
- test set, MST, lambda: 0.75, 'uas': 0.08874492090681528, 'las': 0.059471667170061346

Overall performance showed relatively low compared to English. This may happen as Korean grammer is generally more complex and hard to design as a dataset. 
Also, the model itself is loaded from the pretrained weight, and might not be well suited to Korean grammer even if the model is finetuned to the Korean dataset. 