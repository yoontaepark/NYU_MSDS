import getpass
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql import Window
import numpy as np
from functools import reduce
from pyspark.sql import DataFrame

def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    # import ratings dataset and convert timestamp datatype from int to time
    ratings = spark.read.options(inferSchema = 'True', header = 'True').csv(f'hdfs:/user/{netID}/movielens/ml-latest-small/ratings.csv')
    ratings = ratings.withColumn("timestamp", from_unixtime(col('timestamp')))
    ratings.printSchema()
    ratings.createOrReplaceTempView('ratings')

#### spliting dataset and save as new csv files 

    #shuffle distinct userIds 
    total_user_list = list(ratings.select('userId').distinct().toPandas()['userId'])
    np.random.seed(0)
    np.random.shuffle(total_user_list)

    #divide 60%, 20%, 20% of each userIds to train, val, test dataset
    train_user = total_user_list[:int(len(total_user_list)*0.6)]
    val_user = total_user_list[int(len(total_user_list)*0.6):int(len(total_user_list)*0.8)]
    test_user = total_user_list[int(len(total_user_list)*0.8):] 

    train_df_temp = ratings.where(ratings.userId.isin(train_user))
    val_df_temp = ratings.where(ratings.userId.isin(val_user))
    test_df_temp = ratings.where(ratings.userId.isin(test_user)) 

    #for val, test dataset, move 60% of old data to train dataset 
    val_df_temp = val_df_temp.withColumn("rank", percent_rank().over(Window.partitionBy("userId").orderBy("timestamp")))
    test_df_temp = test_df_temp.withColumn("rank", percent_rank().over(Window.partitionBy("userId").orderBy("timestamp")))

    train_df_from_val = val_df_temp.where("rank <= 0.6").drop("rank")
    val_df = val_df_temp.where("rank > 0.6").drop("rank")

    train_df_from_test = test_df_temp.where("rank <= 0.6").drop("rank")
    test_df = test_df_temp.where("rank > 0.6").drop("rank")

    #for moved data, merge into train dataset
    train_df = reduce(DataFrame.unionAll, [train_df_temp, train_df_from_val, train_df_from_test])

    #save as csv files
    train_df.coalesce(1).write.option('header', True).csv('train_df_large.csv', mode='overwrite')
    val_df.coalesce(1).write.option('header', True).csv('val_df_large.csv', mode='overwrite')
    test_df.coalesce(1).write.option('header', True).csv('test_df_large.csv', mode='overwrite')


if __name__ == "__main__":

    spark = SparkSession.builder.appName('create_csv').getOrCreate()
    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
