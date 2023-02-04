# Lab 1 results

Your name: Yoon Tae Park 

## Part 1
Paste the results of your queries for each question given in the README below:

1.
(262, 'Shawn Greenlee')
(6409, 'The Green Kingdom')
(18361, 'Veronica Green')
(20122, 'Georgia & August Greenberg')

2. 
('Album',)
('Live Performance',)
('Single Tracks',)
('Radio Program',)

3. (13010, 'Entries')

4. (244,)

5. 
('de',)
('en',)
('es',)
('fr',)
('it',)
('pt',)
('ru',)
('sr',)
('tr',)

6. (200,)

7. (158,)

8. (15891, 'Kosta T')

## Part 2

- Execution time before optimization:
Mean time: 0.053 [seconds/query]
Best time: 0.011 [seconds/query]

- Execution time after optimization:
Mean time: 0.023 [seconds/query]
Best time: 0.005 [seconds/query]

- Briefly describe how you optimized for this query:
Best practice) Indexing on the same first column, but calling that column from linked table (track)
-> Result: Positive, performance has greatly improved. This may happen since we are indexing the column that actually can work as a primary key (so, it can distinctly classify each rows)
(code: cursor.execute('create index track_id_idx on track(artist_id)')) 

- Did you try anything other approaches?  How did they compare to your final answer?
 * Second trial) Indexing on the aggregate function (which is track.id)
-> Result: Neutral, performance didn't changed. This may happen since we are not actually using this index as this column is already used for aggregation.

- Before optimization:
Mean time: 0.051 [seconds/query]
Best time: 0.010 [seconds/query]
 - After optimization:
Mean time: 0.051 [seconds/query]
Best time   : 0.010 [seconds/query]
(code: cursor.execute('create index track_id_idx_2 on track(id)’))

 * Third trial) Indexing on multiple columns, so in this case, indexing on best practice + second trial
-> Result: Positive, performance has greatly improved. Due to the effect of best practice, performance has greatly improved. Also, we may assume that when we do multiple indexing, it sums the effect of each indexing methods.  

- Before optimization:
Mean time: 0.052 [seconds/query]
Best time: 0.010 [seconds/query]
- After optimization:
Mean time: 0.023 [seconds/query]
Best time: 0.005 [seconds/query]
(code: cursor.execute('create index track_id_idx on track(artist_id)')
       cursor.execute('create index track_id_idx_2 on track(id)'))

* Fourth trial) Indexing on column that is not actually an index column (ex. artist(artist_active_year_begin))
-> Result: Neutral, performance didn't changed. This may happen since even this column is in the dataset, we are not using this column as an index. 
- Before optimization:
Mean time: 0.053 [seconds/query]
Best time: 0.011 [seconds/query]
- After optimization:
Mean time: 0.053 [seconds/query]
Best time   : 0.011 [seconds/query]
(code: cursor.execute('create index track_id_idx_3 on artist(artist_active_year_begin)'))


