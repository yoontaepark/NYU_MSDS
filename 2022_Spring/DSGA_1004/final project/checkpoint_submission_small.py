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

def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    print('Final Project dataframe loading')

    # import ratings dataset and convert timestamp datatype from int to time
    ratings = spark.read.options(inferSchema = 'True', header = 'True').csv(f'hdfs:/user/{netID}/movielens/ml-latest-small/ratings.csv')
    ratings = ratings.withColumn("timestamp", from_unixtime(col('timestamp')))
    ratings.printSchema()
    ratings.createOrReplaceTempView('ratings')

#### Basic recommender system (80% of grade)
# 1. As a first step, you will need to partition the rating data into training, validation, and test samples as discussed in lecture. 
# I recommend writing a script do this in advance, and saving the partitioned data for future use. 
# This will reduce the complexity of your experiment code down the line, and make it easier to generate alternative splits 
# if you want to measure the stability of your implementation.

    # split df into train/validation/test sets.
    # Dividing oldest 60% to train, and split randomly among remaining 40% of dataset into validation(20%), test(20%) sets.
    ratings = ratings.withColumn("rank", percent_rank().over(Window.partitionBy("userId").orderBy("timestamp")))
    
    train_df = ratings.where("rank <= 0.6").drop("rank")
    val_test_df = ratings.where("rank > 0.6").drop("rank")
    
    userId_median = val_test_df.approxQuantile("userId", [0.5], 0)
    val_df = val_test_df.where(val_test_df.userId <= userId_median[0])    
    test_df = val_test_df.subtract(val_df)

#### 2. Before implementing a sophisticated model, you should begin with a popularity baseline model as discussed in class. 
# This should be simple enough to implement with some basic dataframe computations. Evaluate your popularity baseline (see below) 
# before moving on to the enxt step.
    
    # # get average rating of each movies in training dataset
    # print('1. Popularity baseline')
    # train_movie_avg = train_df.groupby('movieId').agg(mean('rating').alias("mean_rating")).sort(desc("mean_rating"))
    # train_movie_100 = list(train_movie_avg.limit(100).select('*').toPandas()['movieId'])
   
    # ## apply to validation set
    # val_user_list = val_df.select('userId').distinct().count()
    # pred_df_by_actual = val_df.orderBy('rating').groupBy('userId').agg(collect_set('movieId').alias('movie_actual_list'))

    # a = list(pred_df_by_actual.select('*').toPandas()['movie_actual_list'])
    # b = list([train_movie_100] * val_user_list)

    # actual_pred = list(zip(a, b))
    # actual_pred = sc.parallelize(actual_pred)
 
    # val_score = RankingMetrics(actual_pred)
    # print('val_score: ', val_score.meanAveragePrecisionAt(100))

    # ## apply to test set
    # test_user_list = test_df.select('userId').distinct().count()
    # pred_df_by_actual = test_df.orderBy('rating').groupBy('userId').agg(collect_set('movieId').alias('movie_actual_list'))

    # a = list(pred_df_by_actual.select('*').toPandas()['movie_actual_list'])
    # b = list([train_movie_100] * test_user_list)

    # actual_pred = list(zip(a, b))
    # actual_pred = sc.parallelize(actual_pred)

    # test_score = RankingMetrics(actual_pred)
    # print('test_score: ', test_score.meanAveragePrecisionAt(100))

#### 3. Your recommendation model should use Spark's alternating least squares (ALS) method to learn latent factor representations for users and items. 
# Be sure to thoroughly read through the documentation on the pyspark.ml.recommendation module before getting started. 
# This model has some hyper-parameters that you should tune to optimize performance on the validation set, notably:
# the rank (dimension) of the latent factors, and the regularization parameter.


    print('2. Recommandation model using ALS method')
    # Build the recommendation model using ALS on the training data
    # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    als = ALS(rank=50, maxIter=10, regParam=0.001, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop", seed=0)
    model = als.fit(train_df)

    #### testing
    #### use df.rdd.map 
    val_df.drop('rating')
    val_df.show()
    
    pred_val_df = model.transform(val_df)

    rating_window = Window.partitionBy("userId").orderBy(desc("rating"))
    pred_df_by_actual = val_df.withColumn("temp_val_1", collect_list("movieId").over(rating_window)).groupBy("userId").agg(max("temp_val_1").alias("movie_actual_list"))

    prediction_window = Window.partitionBy("userId").orderBy(desc("prediction"))
    pred_df_by_als = pred_val_df.withColumn("temp_val_2", collect_list("movieId").over(prediction_window)).groupBy("userId").agg(max("temp_val_2").alias("movie_pred_list"))

    actual_pred_movie_df = pred_df_by_als.join(pred_df_by_actual, 'userId', 'inner')
    actual_pred = actual_pred_movie_df.rdd.map(lambda x: (x.movie_pred_list, x.movie_actual_list))

    val_score = RankingMetrics(actual_pred)
    print('val_score: ', val_score.meanAveragePrecisionAt(100))




    # print('2. Recommandation model using ALS method') 
    # # Build the recommendation model using ALS on the training data
    # # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    # als = ALS(maxIter=5, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
    # model = als.fit(train_df)
    
    # ## valiation data
    # ## predict the movie rating on the val data
    # pred_val_df = model.transform(val_df)

    # pred_df_by_actual = val_df.orderBy('rating').groupBy('userId').agg(collect_set('movieId').alias('movie_actual_list')).sort(desc('userId'))
    # pred_df_by_als = pred_val_df.orderBy('prediction').groupBy('userId').agg(collect_set('movieId').alias('movie_pred_list')).sort(desc('userId'))
    # actual_pred_movie_df = pred_df_by_actual.join(pred_df_by_als, pred_df_by_actual.userId == pred_df_by_als.userId, 'inner')





    # a = list(actual_pred_movie_df.select('*').toPandas()['movie_actual_list']) 
    # b = list(actual_pred_movie_df.select('*').toPandas()['movie_pred_list']) 

    # actual_pred = list(zip(a, b))
    # actual_pred = sc.parallelize(actual_pred)

    val_score = RankingMetrics(actual_pred)
    print('val_score: ', val_score.meanAveragePrecisionAt(100))

    # ## test data 
    # ## predict the movie rating on the test data
    # pred_test_df = model.transform(test_df)

    # pred_df_by_actual = test_df.orderBy('rating').groupBy('userId').agg(collect_set('movieId').alias('movie_actual_list')).sort(desc('userId'))
    # pred_df_by_als = pred_test_df.orderBy('prediction').groupBy('userId').agg(collect_set('movieId').alias('movie_pred_list')).sort(desc('userId'))
    # actual_pred_movie_df = pred_df_by_actual.join(pred_df_by_als, pred_df_by_actual.userId == pred_df_by_als.userId, 'inner')

    # a = list(actual_pred_movie_df.select('*').toPandas()['movie_actual_list'])
    # b = list(actual_pred_movie_df.select('*').toPandas()['movie_pred_list'])

    # actual_pred = list(zip(a, b))
    # actual_pred = sc.parallelize(actual_pred)

    # test_score = RankingMetrics(actual_pred)
    # print('test_score: ', test_score.meanAveragePrecisionAt(100))


if __name__ == "__main__":

    spark = SparkSession.builder.appName('final_project').getOrCreate()
    sc = spark.sparkContext 
    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
