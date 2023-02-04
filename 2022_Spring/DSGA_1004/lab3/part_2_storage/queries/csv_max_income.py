#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Python script to run benchmark on a query with a file path.
Usage:
    $ spark-submit csv_max_income.py <file_path>
'''


# Import command line arguments and helper functions
import sys
import bench
import statistics

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession


def csv_max_income(spark, file_path):
    '''Construct a basic query on the people dataset

    This function returns a uncomputed dataframe that
    will compute the maximum income grouped by last name

    Parameters
    ----------
    spark : spark session object

    file_path : string
        The path (in HDFS) to the CSV file, e.g.,
        `hdfs:/user/bm106/pub/people_small.csv`

    Returns
    df_max_income:
        Uncomputed dataframe of the maximum income grouped by last_name
    '''

    #TODO
    # Contains csv_max_income function that returns a DataFrame which computes the maximum income grouped by last_name.
    people = spark.read.csv(file_path, header=True, 
                            schema='first_name STRING, last_name STRING, income FLOAT, zipcode INT')

    people.createOrReplaceTempView('people')

    df_max_income = spark.sql('select last_name, max(income) from people group by last_name')
    
    return df_max_income



def main(spark, file_path):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    which_dataset : string, size of dataset to be analyzed
    '''
    #TODO
    times = bench.benchmark(spark, 25, csv_max_income, file_path)

    print(f'Times to run Basic Query 25 times on {file_path}')
    print(times)

    print(f'Minimum Time taken to run Basic Query 25 times on {file_path}:{min(times)}')
    print(f'Median Time taken to run Basic Query 25 times on {file_path}:{statistics.median(times)}')
    print(f'Maximum Time taken to run Basic Query 25 times on {file_path}:{max(times)}')



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part2').getOrCreate()

    # Get file_path for dataset to analyze
    file_path = sys.argv[1]

    main(spark, file_path)
