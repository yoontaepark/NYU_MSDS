import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

from pyspark.sql.functions import *
from datetime import datetime

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.evaluation import RankingEvaluator

from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql import Window

## adding evaludation metric
from pyspark.mllib.evaluation import RankingMetrics
from pyspark import SparkContext

## adding more library
import numpy as np
from functools import reduce
from pyspark.sql import DataFrame
import time


def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    print('Final Project dataframe loading')

    # import ratings dataset and convert timestamp datatype from int to time
    ratings = spark.read.options(inferSchema = 'True', header = 'True').csv(f'hdfs:/user/{netID}/movielens/ml-latest/ratings.csv')
    ratings = ratings.withColumn("timestamp", from_unixtime(col('timestamp')))
    ratings.printSchema()
    ratings.createOrReplaceTempView('ratings')

#### Basic recommender system (80% of grade)
# 1. As a first step, you will need to partition the rating data into training, validation, and test samples as discussed in lecture. 
# I recommend writing a script do this in advance, and saving the partitioned data for future use. 
# This will reduce the complexity of your experiment code down the line, and make it easier to generate alternative splits 
# if you want to measure the stability of your implementation.

    # shuffle distinct userIds 
    total_user_list = list(ratings.select('userId').distinct().toPandas()['userId'])
    np.random.seed(0)
    np.random.shuffle(total_user_list)

    # divide 60%, 20%, 20% of each userIds to train, val, test dataset
    train_user = total_user_list[:int(len(total_user_list)*0.6)]
    val_user = total_user_list[int(len(total_user_list)*0.6):int(len(total_user_list)*0.8)]
    test_user = total_user_list[int(len(total_user_list)*0.8):]

    train_df_temp = ratings.where(ratings.userId.isin(train_user))
    val_df_temp = ratings.where(ratings.userId.isin(val_user))
    test_df_temp = ratings.where(ratings.userId.isin(test_user))

    # for val, test dataset, move 60% of old data to train dataset 
    val_df_temp = val_df_temp.withColumn("rank", percent_rank().over(Window.partitionBy("userId").orderBy("timestamp")))
    test_df_temp = test_df_temp.withColumn("rank", percent_rank().over(Window.partitionBy("userId").orderBy("timestamp")))

    train_df_from_val = val_df_temp.where("rank <= 0.6").drop("rank")
    val_df = val_df_temp.where("rank > 0.6").drop("rank")

    train_df_from_test = test_df_temp.where("rank <= 0.6").drop("rank")
    test_df = test_df_temp.where("rank > 0.6").drop("rank")

    # for moved data, merge into train dataset
    train_df = reduce(DataFrame.unionAll, [train_df_temp, train_df_from_val, train_df_from_test])

    # save as csv files, this will be used in extensiom or local version
    # train_df.coalesce(1).write.option('header', True).csv('train_df_large.csv')
    # val_df.coalesce(1).write.option('header', True).csv('val_df_large.csv')
    # test_df.coalesce(1).write.option('header', True).csv('test_df_large.csv')

#### 2. Before implementing a sophisticated model, you should begin with a popularity baseline model as discussed in class. 
# This should be simple enough to implement with some basic dataframe computations. Evaluate your popularity baseline (see below) 
# before moving on to the enxt step.

    # get average rating of each movies in training dataset
    print('1. Popularity baseline')
    train_movie_avg = train_df.groupby('movieId').agg(mean('rating').alias("mean_rating")).sort(desc("mean_rating"))
    train_movie_100 = list(train_movie_avg.limit(100).select('*').toPandas()['movieId'])

    ## apply to validation set
    val_user_list = val_df.select('userId').distinct().count()
    pred_df_by_actual = val_df.orderBy('rating').groupBy('userId').agg(collect_set('movieId').alias('movie_actual_list'))

    a = list([train_movie_100] * val_user_list)
    b = list(pred_df_by_actual.select('movie_actual_list').toPandas()['movie_actual_list'])

    actual_pred = list(zip(a, b))
    actual_pred = sc.parallelize(actual_pred)

    val_score = RankingMetrics(actual_pred)
    print('val_score: ', val_score.meanAveragePrecisionAt(100))

    ## apply to test set
    test_user_list = test_df.select('userId').distinct().count()
    pred_df_by_actual = test_df.orderBy('rating').groupBy('userId').agg(collect_set('movieId').alias('movie_actual_list'))

    a = list([train_movie_100] * test_user_list)
    b = list(pred_df_by_actual.select('movie_actual_list').toPandas()['movie_actual_list'])

    actual_pred = list(zip(a, b))
    actual_pred = sc.parallelize(actual_pred)

    test_score = RankingMetrics(actual_pred)
    print('test_score: ', test_score.meanAveragePrecisionAt(100))

#### 3. Your recommendation model should use Spark's alternating least squares (ALS) method to learn latent factor representations for users and items. 
# Be sure to thoroughly read through the documentation on the pyspark.ml.recommendation module before getting started. 
# This model has some hyper-parameters that you should tune to optimize performance on the validation set, notably:
# the rank (dimension) of the latent factors, and the regularization parameter.

    print('2. Recommandation model using ALS method')
    # Build the recommendation model using ALS on the training data
    # find best parameter and update
    als = ALS(rank=50, maxIter=10, regParam=0.001, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop", seed=0)

    # measure training time (fit)
    start_time = time.time()
    model = als.fit(train_df)
    end_time = time.time()
    print("Total execution time: {} seconds".format(end_time - start_time))

    # prediction on validation set
    val_df_dropped = val_df.drop('rating', 'timestamp')
    model.transform(val_df_dropped)
    user_recs = model.recommendForAllUsers(100)

    pred_df_by_als = user_recs.withColumn('movie_pred_list', user_recs.recommendations.movieId)

    rating_window = Window.partitionBy("userId").orderBy(desc("rating"))
    pred_df_by_actual = val_df.withColumn("temp_val_1", collect_list("movieId").over(rating_window)).groupBy("userId").agg(max("temp_val_1").alias("movie_actual_list"))

    actual_pred_movie_df = pred_df_by_als.join(pred_df_by_actual, 'userId', 'inner')
    actual_pred = actual_pred_movie_df.rdd.map(lambda x: (x.movie_pred_list, x.movie_actual_list))

    val_score = RankingMetrics(actual_pred)
    print('val_score: ', val_score.meanAveragePrecisionAt(100))

    # prediction on test set
    test_df_dropped = test_df.drop('rating', 'timestamp')
    model.transform(test_df_dropped)
    user_recs = model.recommendForAllUsers(100)

    pred_df_by_als = user_recs.withColumn('movie_pred_list', user_recs.recommendations.movieId)

    rating_window = Window.partitionBy("userId").orderBy(desc("rating"))
    pred_df_by_actual = test_df.withColumn("temp_val_1", collect_list("movieId").over(rating_window)).groupBy("userId").agg(max("temp_val_1").alias("movie_actual_list"))

    actual_pred_movie_df = pred_df_by_als.join(pred_df_by_actual, 'userId', 'inner')
    actual_pred = actual_pred_movie_df.rdd.map(lambda x: (x.movie_pred_list, x.movie_actual_list))

    test_score = RankingMetrics(actual_pred)
    print('test_score: ', test_score.meanAveragePrecisionAt(100))


if __name__ == "__main__":

    spark = SparkSession.builder.appName('final_project').getOrCreate()
    sc = spark.sparkContext
    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)