# Lab 4: Dask

- Name: Yoon Tae Park

- NetID: yp2201

## Description of your solution
#### - What strategies did you use to optimize the computation for the full set?

```
1. Using filter at bag, and convert to dataframe
First of all, I did tried using filter after converting to dataframe. However, it didn't work in 'all' dataset. Considering general tips, I've filter data in bag first. 
ex)
files_all_bags_flatted = files_all_bags_flatted.filter(lambda x : (x['value'] != -9999) & (x['quality']== ' '))
files_all_bags_flatted_prcp = files_all_bags_flatted.filter(lambda x : x['element'] == 'PRCP')
files_all_bags_flatted_tmax = files_all_bags_flatted.filter(lambda x : x['element'] == 'TMAX')

However, while this worked for 'tiny', and 'small' dataset, this method didn't work in 'all' dataset. 


2. Using client.persist before .compute()
I've tried to use .persist before .compute(). This makes computation more faster(in tiny, small dataset), but still didn't work in 'all' dataset. 


3. Using delay object before .compute() 
I've tried to use delay object before .compute(). I've assumed that converting bag -> dataframe -> delay -> dataframe should make some progress.(especially for .compute())
It made some optimized result for 'tiny', 'small' computation
However, it seems that more methods are needed make 'all' computation work


4. Defining columns when converting into dataframe
When converting into dataframe, I don't need irrelevant columns, so I pre-defined columns when converting into dataframe. 
This made 'all' computation work, and the result follows as below:

- all_prcp) CPU times: user 4.43 ms, sys: 1.06 ms, total: 5.49 ms, Wall time: 8.48 ms 
- all_tmax) CPU times: user 4.4 ms, sys: 1.02 ms, total: 5.42 ms, Wall time: 17.1 ms


5. Using meta instead of column
When converting into dataframe, I've tried to use meta, which pre-defines columns with its data type. 
This made some optimization, and the result follows as below:

- all_prcp) CPU times: user 5.36 ms, sys: 53 µs, total: 5.42 ms, Wall time: 11.2 ms
- all_tmax) CPU times: user 5.12 ms, sys: 0 ns, total: 5.12 ms, Wall time: 11.3 ms


6. Sort after compute 
I've tried to sort result after compute, as sorting itself needs extra computation. 
This made some optimization, and the result follows as below:

- all_prcp) CPU times: user 4.4 ms, sys: 892 µs, total: 5.29 ms, Wall time: 9.12 ms
- all_tmax) CPU times: user 4.96 ms, sys: 0 ns, total: 4.96 ms, Wall time: 8.66 ms


7. Setting npartitions to 10,000
As default npartition is 101, I've increased npartition to 10,000 as it has a huge dataset
This made an optimizated result(final result)

- all_prcp) CPU times: user 3.67 ms, sys: 1.01 ms, total: 4.69 ms, Wall time: 8.25 ms
- all_tmax) CPU times: user 4.77 ms, sys: 23 µs, total: 4.79 ms, Wall time: 8.09 ms


8. Setting repartition to 10,000
This method didn't worked, and maybe this may happened since partitions are already set to 10,000
- all_prcp) CPU times: user 13 s, sys: 452 ms, total: 13.4 s, Wall time: 22.6 s
- all_tmax) CPU times: user 15.1 s, sys: 408 ms, total: 15.5 s, Wall time: 23.9 s


```


#### - Did you try any alternative implementations that didn't work, or didn't work as well?  If so, how did this change your approach?

```
1. Using filter at bag, and convert to dataframe
2. Using client.persist before .compute()
3. Using delay object before .compute() 
4. Setting repartition to 10,000

Above methods worked well for 'tiny' and 'small' dataset. However, above methods didn't worked well for 'all' dataset. 
Especially 1~3 approach, it failed to compute the result, so I've changed my approach from applying a single approach to applying multiple approaches.

For approach 4, repartition didn't worked, and I've decided to exclude this approach.


```


#### - Did you encounter any unexpected behavior?

```
Especially, handling 'all' data mostly had failure, compared to handling 'tiny', and 'small'. 
So I've changed my cluster settings to upper limit. 

Also, as I've tried to use some optimization methods, it failed to work in 'all' dataset. 
So it was hard to guess if certain methods are part of optimal methods, or if certain methods are not optimal methods at all.

For example, repartition method for 'all' dataset was not optimizing the result, but it may not optimized the result since this method was applied at the very last trial. 
(So, some of the methods already optimized the result, and that is why this method didn't worked) 


```