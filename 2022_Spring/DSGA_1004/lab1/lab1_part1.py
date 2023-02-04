#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# USAGE:
#   python lab1_part1.py music_small.db

import sys
import sqlite3


# The database file should be given as the first argument on the command line
# Please do not hard code the database file!
db_file = sys.argv[1]

# We connect to the database using 
with sqlite3.connect(db_file) as conn:
    # This query counts the number of artists who became active in 1990
    # year = (1990,)
    # for row in conn.execute('SELECT count(*) FROM artist WHERE artist_active_year_begin=?', year):
    #     # Since there is no grouping here, the aggregation is over all rows
    #     # and there will only be one output row from the query, which we can
    #     # print as follows:
    #     print('Tracks from {}: {}'.format(year[0], row[0]))
        
        # The [0] bits here tell us to pull the first column out of the 'year' tuple
        # and query results, respectively.

    # ADD YOUR CODE STARTING HERE
    
    # Question 1
    print('Question 1:')
    
    # implement your solution to q1
    for row in conn.execute('SELECT id, artist_name FROM artist WHERE lower(artist_name) like "%green%"'):
        print(row)
    print('---')
    
    # Question 2
    print('Question 2:')
    
    # implement your solution to q2
    for row in conn.execute('SELECT DISTINCT album_type FROM album WHERE album_type IS NOT NULL'):
        print(row)
    print('---')
    
    # Question 3
    print('Question 3:')
    
    # implement your solution to q3
    for row in conn.execute('SELECT id, album_title from album ORDER BY album_listens DESC LIMIT 1'):
        print(row)    
    print('---')
    
    # Question 4
    print('Question 4:')
    
    # implement your solution to q4
    # Assumption: if wikipedia page is not null, then we can see that as a wikipedia page
    for row in conn.execute('SELECT count(DISTINCT artist_name) FROM artist WHERE artist_wikipedia_page IS NOT NULL'):
        print(row)    
    print('---')
    
    # Question 5
    print('Question 5:')
    
    # implement your solution to q5
    for row in conn.execute('SELECT track_language_code from track WHERE track_language_code IS NOT NULL GROUP BY track_language_code HAVING count(id) >= 3'):
        print(row)    
    print('---')
    
    # Question 6
    print('Question 6:')
    
    # implement your solution to q6
    for row in conn.execute('SELECT count(DISTINCT track.id) FROM  track INNER JOIN artist ON track.artist_id = artist.id where artist.artist_latitude < 0'):
        print(row)    
    print('---')
    
    # Question 7
    print('Question 7:')
    
    # implement your solution to q7
    for row in conn.execute('SELECT count(DISTINCT album.id) FROM track INNER JOIN artist ON track.artist_id = artist.id INNER JOIN album ON track.album_id = album.id WHERE album.album_title = artist.artist_name'):
        print(row)    
    print('---')
    
    # Question 8
    print('Question 8:')
    
    # implement your solution to q8
    for row in conn.execute('SELECT artist.id, artist.artist_name FROM track INNER JOIN artist ON track.artist_id = artist.id INNER JOIN album ON track.album_id = album.id GROUP BY artist.id, artist.artist_name ORDER BY count(DISTINCT album.id) DESC limit 1'):
        print(row)    
    print('---')
