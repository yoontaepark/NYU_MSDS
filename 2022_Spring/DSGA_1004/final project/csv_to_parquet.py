import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

def main(spark, netID):
    '''Main routine for run for Storage optimization template.
    Parameters
    ----------
    spark : SparkSession object

    '''
    ratings = spark.read.options(inferSchema = 'True', header = 'True').csv(f'hdfs:/user/{netID}/movielens/ml-latest/ratings.csv')
    ratings = ratings.withColumn("timestamp", from_unixtime(col('timestamp')))

    ratings = ratings.repartition(100000, 'userId')
    ratings.write.parquet('ratings_100000.parquet')


# Only enter this block if we're in main
if __name__ == "__main__":

    spark = SparkSession.builder.appName('csv_to_parquet').getOrCreate()
    sc=spark.sparkContext 
    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
