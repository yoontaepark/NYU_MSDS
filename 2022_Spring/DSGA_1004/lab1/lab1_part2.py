#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# USAGE:
#   python lab1_part2.py music_small.db

import sys
import sqlite3
import timeit


# The database file should be given as the first argument on the command line
# Please do not hard code the database file!
db_file = sys.argv[1]


# The query to be optimized is given here
MY_QUERY = """SELECT artist.*, count(distinct track.id) as tracks
                FROM track INNER JOIN artist
                ON track.artist_id = artist.id
                GROUP BY artist_id
                HAVING tracks > 10"""

NUM_ITERATIONS = 100

def run_my_query(conn):
    for row in conn.execute(MY_QUERY):
        pass
    


# We connect to the database using
with sqlite3.connect(db_file) as conn:
    # We use a "cursor" to mark our place in the database.
    cursor = conn.cursor()

    # We could use multiple cursors to keep track of multiple
    # queries simultaneously.

    # drop index for iteration
    # cursor.execute('drop index track_id_idx')
    # cursor.execute('drop index track_id_idx_2')
    # cursor.execute('drop index track_id_idx_3')
    
    
    orig_time = timeit.repeat('run_my_query(conn)', globals=globals(), number=NUM_ITERATIONS)
    print("Before optimization:")
    
    print(f'Mean time: {sum(orig_time)/NUM_ITERATIONS:.3f} [seconds/query]')
    print(f'Best time: {min(orig_time)/NUM_ITERATIONS:.3f} [seconds/query]')

    # MAKE YOUR MODIFICATIONS TO THE DATABASE HERE
    # Best practice) Indexing on the same first column, but calling that column from linked table (track)
    # Before optimization:
    # Mean time: 0.053 [seconds/query]
    # Best time: 0.011 [seconds/query]
    # After optimization:
    # Mean time: 0.023 [seconds/query]
    # Best time   : 0.005 [seconds/query]
    cursor.execute('create index track_id_idx on track(artist_id)') 

    # Second trial) Indexing on the aggregate function (which is track.id)
    # Before optimization:
    # Mean time: 0.051 [seconds/query]
    # Best time: 0.010 [seconds/query]
    # After optimization:
    # Mean time: 0.051 [seconds/query]
    # Best time   : 0.010 [seconds/query]
    # cursor.execute('create index track_id_idx_2 on track(id)')

    
    # Thrid trial) Indexing on multiple columns, so in this case, indexing on best practice + second trial
    # Before optimization:
    # Mean time: 0.052 [seconds/query]
    # Best time: 0.010 [seconds/query]
    # After optimization:
    # Mean time: 0.023 [seconds/query]
    # Best time   : 0.005 [seconds/query]
    # cursor.execute('create index track_id_idx on track(artist_id)')
    # cursor.execute('create index track_id_idx_2 on track(id)')

    # Fourth trial) Indexing on column that is not acutally an index column (ex. artist(artist_active_year_begin))
    # cursor.execute('create index track_id_idx_3 on artist(artist_active_year_begin)')
    # Before optimization:
    # Mean time: 0.053 [seconds/query]
    # Best time: 0.011 [seconds/query]
    # After optimization:
    # Mean time: 0.053 [seconds/query]
    # Best time   : 0.011 [seconds/query]


    new_time = timeit.repeat('run_my_query(conn)', globals=globals(), number=NUM_ITERATIONS)
    print("After optimization:")

    print(f'Mean time: {sum(new_time)/NUM_ITERATIONS:.3f} [seconds/query]')
    print(f'Best time   : {min(new_time)/NUM_ITERATIONS:.3f} [seconds/query]')
