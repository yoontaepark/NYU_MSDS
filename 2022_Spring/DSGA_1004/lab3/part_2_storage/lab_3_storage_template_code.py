#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Template script to connect to Active Spark Session
Usage:
    $ spark-submit lab_3_storage_template_code.py <any arguments you wish to add>
'''


# Import command line arguments and helper functions(if necessary)
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession



def main(spark):
    '''Main routine for run for Storage optimization template.
    Parameters
    ----------
    spark : SparkSession object

    '''
    #####--------------YOUR CODE STARTS HERE--------------#####

    #Use this template to as much as you want for your parquet saving and optimizations!
    ### 2.5: trial 1) Sort the DataFrame according to particular columns before writing out to parquet.
    # ### people_small 
    # people_small = spark.read.csv('hdfs:/user/bm106/pub/people_small.csv', header=True, 
    #                     schema='first_name STRING, last_name STRING, income FLOAT, zipcode INT')
    # # people_small.write.parquet('people_small.parquet')

    ### people_large 
    # people_large = spark.read.csv('hdfs:/user/bm106/pub/people_large.csv', header=True, 
    #                     schema='first_name STRING, last_name STRING, income FLOAT, zipcode INT')

    # people_large.write.parquet('people_large.parquet')

    # ## 2.5) trial 1: Sort the DataFrame according to particular columns before writing out to parquet.
    # ### this code block is for 2.5: trial 1 
    # ### will try by income
    # # people.sort_values(by='income', inplace=True)
    # people_small.sort('income', inplace=True)
    # # .parquet_income: sort by income     
    # people_small.write.parquet('people_small_income.parquet')


    
    # ### people_medium
    # people_medium = spark.read.csv('hdfs:/user/bm106/pub/people_medium.csv', header=True, 
    #                     schema='first_name STRING, last_name STRING, income FLOAT, zipcode INT')
    # people_medium.sort('income', inplace=True)
    # people_medium.write.parquet('people_medium_income.parquet')

    # ### people_large
    # people_large = spark.read.csv('hdfs:/user/bm106/pub/people_large.csv', header=True, 
    #                     schema='first_name STRING, last_name STRING, income FLOAT, zipcode INT')
    # people_large.sort('income', inplace=True)
    # people_large.write.parquet('people_large_income.parquet')   

    ### 2.5: trial 2) Change the partition structure of the data.
    ### people_small 
    # people_small = spark.read.csv('hdfs:/user/bm106/pub/people_small.csv', header=True, 
    #                     schema='first_name STRING, last_name STRING, income FLOAT, zipcode INT')
    # people_small = people_small.repartition(10, 'zipcode')
    # # people_small.explain()
    # people_small.write.parquet('people_small_repartition.parquet')

    # ### people_medium
    # people_medium = spark.read.csv('hdfs:/user/bm106/pub/people_medium.csv', header=True, 
    #                     schema='first_name STRING, last_name STRING, income FLOAT, zipcode INT')
    # # people_medium = people_medium.repartition(10, 'zipcode')
    # people_medium.write.parquet('people_medium_repartition.parquet')

    ### people_large
    # people_large = spark.read.csv('hdfs:/user/bm106/pub/people_large.csv', header=True, 
    #                     schema='first_name STRING, last_name STRING, income FLOAT, zipcode INT')
    # people_large = people_large.repartition(10, 'zipcode')
    # people_large.write.parquet('people_large_repartition.parquet')




# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part2').getOrCreate()

    #If you wish to command line arguments, look into the sys library(primarily sys.argv)
    #Details are here: https://docs.python.org/3/library/sys.html
    #If using command line arguments, be sure to add them to main function

    main(spark)
    