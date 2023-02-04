# Lab 3: Spark and Parquet Optimization Report

Name: Yoon Tae Park
 
NetID: yp2201@nyu.edu

## Part 1: Spark

#### Question 1: 
How would you express the following computation using SQL instead of the object interface? sailors.filter(sailors.age > 40).select(sailors.sid, sailors.sname, sailors.age)

Code:
```SQL

spark.sql('select sid, sname, age from sailors where age > 40')

```


Output:
```

+---+-------+----+
|sid|  sname| age|
+---+-------+----+
| 22|dusting|45.0|
| 31| lubber|55.5|
| 95|    bob|63.5|
+---+-------+----+

```


#### Question 2: 
How would you express the following using the object interface instead of SQL? spark.sql('SELECT sid, COUNT(bid) from reserves WHERE bid != 101 GROUP BY sid')

Code:
```python

reserves.filter(reserves.bid != 101).groupby('sid').agg({'bid' : 'count'})

```


Output:
```

+---+----------+
|sid|count(bid)|
+---+----------+
| 22|         3|
| 31|         3|
| 74|         1|
| 64|         1|
+---+----------+

```

#### Question 3: 
Using a single SQL query, how many distinct boats did each sailor reserve? The resulting DataFrame should include the sailor's id, name, and the count of distinct boats. (Hint: you may need to use first(...) aggregation function on some columns.) Provide both your query and the resulting DataFrame in your response to this question.

Code:
```SQL

spark.sql('select sailors.sid, sailors.sname, count(distinct reserves.bid) from sailors full outer join reserves on sailors.sid = reserves.sid group by sailors.sid, sailors.sname')

```


Output:
```

+---+-------+-------------------+
|sid|  sname|count(DISTINCT bid)|
+---+-------+-------------------+
| 58|  rusty|                  0|
| 64|horatio|                  2|
| 29| brutus|                  0|
| 22|dusting|                  4|
| 31| lubber|                  3|
| 71|  zorba|                  0|
| 85|    art|                  0|
| 74|horatio|                  1|
| 95|    bob|                  0|
| 32|   andy|                  0|
+---+-------+-------------------+


```

#### Question 4: 
Implement a query using Spark transformations which finds for each artist term, compute the median year of release, maximum track duration, and the total number of artists for that term (by ID). What are the results for the ten terms with the shortest average track durations? Include both your query code and resulting DataFrame in your response.

Code:
```python

spark.sql('select artist_term.term, percentile(tracks.year, 0.5), max(tracks.duration), count(artist_term.artistID) from artist_term inner join tracks on artist_term.artistID = tracks.artistID group by artist_term.term order by avg(tracks.duration) asc limit 10')

```


Output:
```

+----------------+----------------------------------------+-------------+---------------+
|            term|percentile(year, CAST(0.5 AS DOUBLE), 1)|max(duration)|count(artistID)|
+----------------+----------------------------------------+-------------+---------------+
|       mope rock|                                     0.0|     13.66159|              1|
|      murder rap|                                     0.0|     15.46404|              1|
|    abstract rap|                                  2000.0|     25.91302|              1|
|experimental rap|                                  2000.0|     25.91302|              1|
|     ghetto rock|                                     0.0|     26.46159|              1|
|  brutal rapcore|                                     0.0|     26.46159|              1|
|     punk styles|                                     0.0|     41.29914|              1|
|     turntablist|                                  1993.0|    145.89342|              7|
| german hardcore|                                     0.0|     45.08689|              1|
|     noise grind|                                  2005.0|     89.80853|              5|
+----------------+----------------------------------------+-------------+---------------+

```
#### Question 5: 
Create a query using Spark transformations that finds the number of distinct tracks associated (through artistID) to each term. Modify this query to return only the top 10 most popular terms, and again for the bottom 10. Include each query and the tables for the top 10 most popular terms and the 10 least popular terms in your response.

##### 10 Most Popular Terms
Code:
```

spark.sql('select artist_term.term, count(distinct tracks.trackID) as cnt from artist_term left join tracks on artist_term.artistID = tracks.artistID group by artist_term.term order by cnt desc limit 10')

```
Output:
```

+----------------+-----+
|            term|  cnt|
+----------------+-----+
|            rock|21796|
|      electronic|17740|
|             pop|17129|
|alternative rock|11402|
|         hip hop|10926|
|            jazz|10714|
|   united states|10345|
|        pop rock| 9236|
|     alternative| 9209|
|           indie| 8569|
+----------------+-----+

```

##### 10 Least Popular Terms
Code:
```

spark.sql('select artist_term.term, count(distinct tracks.trackID) as cnt from artist_term left join tracks on artist_term.artistID = tracks.artistID group by artist_term.term order by cnt limit 10')

```

Output:
```

+--------------------+---+
|                term|cnt|
+--------------------+---+
|    shemale vocalist|  0|
|       boeren muziek|  0|
| psychedelic country|  0|
|           metalgaze|  0|
|           polyphony|  0|
|     milled pavement|  0|
|       salsa boricua|  0|
| finnish death metal|  0|
|        21st century|  0|
|russian drum and ...|  0|
+--------------------+---+

```

## Part 2: Parquet Optimization:

What to include in your report:
  - Tables of all numerical results (min, max, median) for each query/size/storage combination for part 2.3, 2.4 and 2.5.
``` 
2.3			
- csv_avg_income			
	people_small	people_medium	people_large
Min	0.63992238	0.692625284	23.04995394
Median	0.795773029	0.809772968	23.40381908
Max	4.551961422	4.676626444	30.16833425
			
- csv_max_income			
	people_small	people_medium	people_large
Min	0.686404943	0.674203873	22.71998882
Median	0.803663969	0.761995792	23.16814256
Max	4.563187122	4.503527403	34.96219397
			
- csv_anna			
	people_small	people_medium	people_large
Min	0.079406738	0.323654413	24.63457036
Median	0.097906828	0.358760595	25.00041461
Max	2.888526678	3.217741013	29.08719254
  
```

``` 
2.4			
- pq_avg_income			
	people_small	people_medium	people_large
Min	0.707796574	3.587165356	5.68302393
Median	0.913724661	4.101938009	5.89331913
Max	3.826096058	6.477219343	9.78316617
			
- pq_max_income			
	people_small	people_medium	people_large
Min	0.658905745	3.544954538	8.869404316
Median	0.763548136	3.944070101	9.743651628
Max	4.660872221	6.444980145	10.88225245
			
- pq_anna			
	people_small	people_medium	people_large
Min	0.088113308	0.108901024	5.46604681
Median	0.121062517	0.151005745	5.873440742
Max	1.806722164	3.012771368	7.080391884
  
```

``` 
2.5: trial 1) sort by income			
- pq_avg_income			
	people_small	people_medium	people_large
Min	0.687492371	3.61274004	5.548138618
Median	0.856104851	4.019694805	5.89900732
Max	4.683032513	6.317535639	8.514662981
			
- pq_max_income			
	people_small	people_medium	people_large
Min	0.636426687	0.753691435	3.849775314
Median	0.748782873	0.920787811	4.051210642
Max	3.674511671	3.75590682	7.143443584
			
- pq_anna			
	people_small	people_medium	people_large
Min	0.085745335	3.63714695	5.506696701
Median	0.105255365	3.892116308	5.892752409
Max	2.852680922	4.818898439	8.940930367
			
			
2.5: trial 2) repartition(10, 'zipcode')			
- pq_avg_income			
	people_small	people_medium	people_large
Min	0.415720463	3.657489061	3.784240961
Median	0.498479843	3.947556257	4.016852379
Max	4.218042612	6.170673609	8.189275026
			
- pq_max_income			
	people_small	people_medium	people_large
Min	0.422834396	0.444555283	6.295645714
Median	0.50124526	0.55484128	6.480703115
Max	4.127403259	4.330115318	8.316463709
			
- pq_anna			
	people_small	people_medium	people_large
Min	0.094033957	0.129532814	4.567357779
Median	0.147799492	0.173019171	4.862128019
Max	3.351253033	2.990556002	6.212363482
			
			
2.5: trial 3) Change the HDFS replication factor: hdfs dfs -setrep -w 1 hdfs:/user/yp2201/filename			
- pq_avg_income			
	people_small	people_medium	people_large
Min	0.724547625	0.449991465	7.796478271
Median	0.863206148	0.547646523	13.79577422
Max	3.997735023	4.327079773	27.69688964
			
- pq_max_income			
	people_small	people_medium	people_large
Min	0.702949524	0.429586887	4.023491144
Median	0.808372259	0.52260828	16.86921883
Max	4.879172087	4.08241272	508.8583548
			
- pq_anna			
	people_small	people_medium	people_large
Min	0.082099438	0.107987165	5.596507549
Median	0.105215311	0.145423651	5.86025548
Max	2.329555511	2.973102331	9.065425634
  
```


  - How do the results in parts 2.3, 2.4, and 2.5 compare?
```
  
    - Comparing 2.3 vs 2.4: Overall, 2.4 performed better. (For all avg_income, max_income, and anna files) 
    Especially for the people_large dataset, 2.4 outperfomed 2.3.
    For people_small, and people_medium dataset, results were slightly better for 2.4. 
    Also for those datasets, there were some results that 2.3 performed better compared to 2.4, but this may happened due to system environment. 
    
    - Comparing 2.3 vs 2.5: Overall, 2.5 performed better. (For all avg_income, max_income, and anna files) 
    Especially for the people_large dataset, 2.5 outperfomed 2.3.
    For people_small, and people_medium dataset, results were slightly better for 2.5. 
    Also for those datasets, there were some results that 2.3 performed better compared to 2.5, but this may happened due to system environment. 
    This result is similar with comparison of 2.3 vs 2.4, as 2.5 is also using .parquet instead of .csv file
  
    - Comparing 2.4 vs 2.5: Depends on the method that I've tried 
    Trial 1: sort by income 
    avg_inocome: Not a big difference. This may occur since getting an average still needs all rows of data to search
    max_income: 2.5 performed better.This may occur since it will only need max value from sorted values
    anna: Not a big difference. This may occur since we are looking both first_name and income condition. Since there were no sort for first_time, getting a result still needs all rows of data to search .
    
    Trial 2: repartition(10, 'zipcode')
    For overall datasets, 2.5 performed better. This shows efficiency of repartition. 

    Trial 3: Change the HDFS replication factor: hdfs dfs -setrep -w 1 hdfs:/user/yp2201/filename
    For small and medium datasets, 2.5 performed better. However, for large dataset, 2.4 performed better. 
    This doesn't guarantee that this method is good enough for optimization and should try different settings to find the optimal. 
    
```
  
  
  - What did you try in part 2.5 to improve performance for each query?
```
  
   Trial 1: sort by income, I've sorted the DataFrame by column 'income' before writing out to parquet.
   
   Trial 2: I've change the partition structure of the data. Especially, I've used repartition(10, 'zipcode') to dataframe and create a new file. 
   
   Trial 3: Change the HDFS replication factor. I've changed HDFS replication factor as 1 (hdfs dfs -setrep -w 1 hdfs:/user/yp2201/filename)
       
```
   
  - What worked, and what didn't work?
```
    
   Trial 1: sort by income
   It worked if the given query don't need to search entire sorted values. For example, it worked in max_income dataset.
   This happens as the query want to get the max_income, and do not need to search every columns.
   However, it didn't work for other queries, as those queries still need to search every row even the dataset is sorted by a certain column.
   
   Trial 2: repartition(10, 'zipcode')
   It worked and showed competitiveness in optimization. Overall showing a great optimization result. 
   
   Trial 3: Change the HDFS replication factor: hdfs dfs -setrep -w 1 hdfs:/user/yp2201/filename  
   It worked for small and medium dataset, but didn't worked for large dataset. This implies that this trial didn't work well for optimization and might try difference condition to find the optimal way. 

```
Basic Markdown Guide: https://www.markdownguide.org/basic-syntax/






