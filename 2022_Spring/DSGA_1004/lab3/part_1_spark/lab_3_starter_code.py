#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py <student_netID>
'''
#Use getpass to obtain user netID
import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession


def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    print('Lab 3 Example dataframe loading and SQL query')

    # Load the boats.txt and sailors.json data into DataFrame
    boats = spark.read.csv(f'hdfs:/user/{netID}/boats.txt')
    sailors = spark.read.json(f'hdfs:/user/{netID}/sailors.json')

    print('Printing boats inferred schema')
    boats.printSchema()
    print('Printing sailors inferred schema')
    sailors.printSchema()
    # Why does sailors already have a specified schema?

    print('Reading boats.txt and specifying schema')
    boats = spark.read.csv('boats.txt', schema='bid INT, bname STRING, color STRING')

    print('Printing boats with specified schema')
    boats.printSchema()

    # Give the dataframe a temporary view so we can run SQL queries
    boats.createOrReplaceTempView('boats')
    sailors.createOrReplaceTempView('sailors')
    # Construct a query
    # print('Example 1: Executing SELECT count(*) FROM boats with SparkSQL')
    # query = spark.sql('SELECT count(*) FROM boats')

    # query.show()

    #####--------------YOUR CODE STARTS HERE--------------#####
    #make sure to load reserves.json, artist_term.csv, and tracks.csv
    #For the CSVs, make sure to specify a schema!

    # Loading datasets
    # question_1_query: already created by variable name sailors 
    # question_2,3 query
    reserves = spark.read.json(f'hdfs:/user/{netID}/reserves.json')
    reserves.createOrReplaceTempView('reserves')

    # question_4,5 query
    artist_term = spark.read.csv(f'hdfs:/user/{netID}/artist_term.csv', schema='artistID STRING, term STRING')
    artist_term.createOrReplaceTempView('artist_term')

    tracks = spark.read.csv(f'hdfs:/user/{netID}/tracks.csv', schema='trackID STRING, title STRING, release STRING, year INT, duration FLOAT, artistID STRING')
    tracks.createOrReplaceTempView('tracks')


    # Question 1: How would you express the following computation using SQL instead of the object interface? 
    # sailors.filter(sailors.age > 40).select(sailors.sid, sailors.sname, sailors.age)
    print('Question 1: How would you express the following computation using SQL instead of the object interface?')
    query_q1 = spark.sql('select sid, sname, age from sailors where age > 40')
    query_q1.show()

    # Question 2: How would you express the following using the object interface instead of SQL? 
    # spark.sql('SELECT sid, COUNT(bid) from reserves WHERE bid != 101 GROUP BY sid')
    print('Question 2: How would you express the following using the object interface instead of SQL?')
    query_q2 = reserves.filter(reserves.bid != 101).groupby('sid').agg({'bid' : 'count'})
    query_q2.show()

    # Question 3: Using a single SQL query, how many distinct boats did each sailor reserve? 
    # The resulting DataFrame should include the sailor's id, name, and the count of distinct boats. 
    # (Hint: you may need to use first(...) aggregation function on some columns.) 
    # Provide both your query and the resulting DataFrame in your response to this question.
    print('Question 3: Using a single SQL query, how many distinct boats did each sailor reserve? ')
    query_q3 = spark.sql('select sailors.sid, sailors.sname, count(distinct reserves.bid) from sailors full outer join reserves on sailors.sid = reserves.sid group by sailors.sid, sailors.sname')
    query_q3.show()

    # Question 4: Implement a query using Spark transformations which finds for each artist term, 
    # compute the median year of release, maximum track duration, and the total number of artists for that term (by ID). 
    # What are the results for the ten terms with the shortest average track durations? 
    # Include both your query code and resulting DataFrame in your response.
    print('Question 4: Implement a query using Spark transformations which finds for each artist term, compute the median year of release, maximum track duration, and the total number of artists for that term (by ID). What are the results for the ten terms with the shortest average track durations?')
    query_q4 = spark.sql('select artist_term.term, percentile(tracks.year, 0.5), max(tracks.duration), count(artist_term.artistID) from artist_term inner join tracks on artist_term.artistID = tracks.artistID group by artist_term.term order by avg(tracks.duration) asc limit 10')
    query_q4.show()      

    # Question 5: Create a query using Spark transformations that finds the number of distinct tracks associated (through artistID) to each term. 
    # Modify this query to return only the top 10 most popular terms, and again for the bottom 10. 
    # Include each query and the tables for the top 10 most popular terms and the 10 least popular terms in your response.
    print('Question 5: Create a query using Spark transformations that finds the number of distinct tracks associated (through artistID) to each term.')
    print('top 10:\n')

    query_q5_top = spark.sql('select artist_term.term, count(distinct tracks.trackID) as cnt from artist_term left join tracks on artist_term.artistID = tracks.artistID group by artist_term.term order by cnt desc limit 10')
    query_q5_top.show()    
 
    print('bottom 10:\n')
    query_q5_bottom = spark.sql('select artist_term.term, count(distinct tracks.trackID) as cnt from artist_term left join tracks on artist_term.artistID = tracks.artistID group by artist_term.term order by cnt limit 10')
    query_q5_bottom.show()    
 

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
