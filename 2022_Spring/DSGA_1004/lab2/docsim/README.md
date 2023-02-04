# Part 5: Document similarity

## Explain each step of your solution
1. MRStep(mapper=self.mapper_get_words, reducer=self.reducer_1):
- mapper=self.mapper_get_words: 
For every line, iterate each word and count as 1. We then yield (doc_name, word) as a key and 1 as a value 
Then, iterate one time more to create a placeholder that count as 0. We then yield (placeholder (i.e. punctuation @)) as a key and 0 as a value

- reducer=self.reducer_1:
Based on mapper key pairs(doc_name, word), sum values. 
Then we re-map word as a key, and (doc_name, sum(values)) as a value

                 
2. MRStep(reducer=self.reducer_2)
- reducer=self.reducer_2:
Based on word key, make a list of values. So values would be list of [(doc_name1, sum(values), doc_name2, sum(values), ...]


3. MRStep(mapper=self.mapper_3, reducer=self.reducer_3)
- mapper=self.mapper_3: 
For each word key, iterate each appended list (doc_name, value) pairs. We need len(pair) * len(pair) combinations, so iterate twice.
So, we now have (doc_i, doc_j, word) as a key, and ((counts for (doc_i, word), (counts for (doc_j and word)) as a value

- reducer=self.reducer_3:
For each (doc_i, doc_j, word) key, select the min value in value list. 


4. MRStep(mapper=self.mapper_4, reducer=self.reducer_4)
- mapper=self.mapper_4: 
For each (doc_i, doc_j, word) key, we don't need word to be yielded.
So only yield (doc_i, doc_j) as a key, and counts as a value

- reducer=self.reducer_4:
Now, for each mapper, for (doc_j, doc_j) key, sum counts and yield (doc_i, doc_j) as a key, and sum(counts) as a value


...
## What problems, if any, did you encounter?
1. Yielding final result with doc_name_i, doc_name_j was a problem. When we just yield word as a key in mapper 1, we should fail to keep doc_names.
So, I've handled this issue with making a key as a (doc_name, word) pair.  

2. Also, we needed a pair of documents. To make this happen, I've made a doc list by word keys. So, for each given word, I've made a list of (doc_name, sum(counts))

3. Creating (documents, word) pair that doesn't have value at all. Basically, we are reading each line and yield given word, so it was hard to think of creating a word with value 0.
This was handled by creating a placeholder with a value 0 when running mapper 1.  I've added one more for loop that created dummy word key with value 0.

4. When using reducer to get min value, the values I've collected was double-listed, so there was error when I don't index the list. This was handled by making values a list, and use index 0, and use min function.

...

## How does your solution scale with the number of documents?
1. For mrstep 1, mapper 1 yields (doc_name, word) key with value 1s. Reducer 1 sums the values given the same key. So, if there were 1000 mappers but can be classified as 10 different keys, mapper results are gathered in 10 reducers, which shows efficiency of parallelism. . If there are more documents, it just classify documents with (doc_name, word) pair, and all values within same key pair goes to same reducer. 

2. For mrstep 2, reducer 2 gathers list of values based on word key. This means that this reducer contains many different key pairs(doc_name, word) in one single word key. If there are more documents, reducer can hold many values into one list for each given word. This gets more effective when there are lots of documents, since some word keys are duplicated for different documents. (i.e. in, it, is, when..) 

3. For the last step(mrstep 4), reducer 4 reduces great amount of rows yielded by mapper 4. Mapper 4 yields key as (doc_i, doc_j) and removes word key. This makes mapper with same key pair(doc_i, doc_j) with lots of different values. Reducer sums those values by key pair(doc_i, doc_j), and we now have combination of two document pairs as a key, with total count values. 

4. Overall, if we just read entire documents, we would get sum(doc_name * word)) rows. By using mappers and reducers, we are able to reduce huge amount of rows and yield rows with combination of two document pairs as a key, and total counts(similarity) as a value.  

...
