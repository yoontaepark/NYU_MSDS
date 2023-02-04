import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

from pyspark.sql.functions import *
from datetime import datetime

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql import Window

def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    print('Final Project dataframe loading')
    

    # rating test(0422)
    ratings = spark.read.options(inferSchema = 'True', header = 'True').csv(f'hdfs:/user/{netID}/movielens/ml-latest-small/ratings.csv')
    ratings.printSchema()       
    ratings = ratings.withColumn("timestamp", from_unixtime(col('timestamp')))

    ratings.printSchema()   
    ratings.createOrReplaceTempView('ratings')

    # split df into training and test sets
    ratings = ratings.withColumn("rank", percent_rank().over(Window.partitionBy().orderBy("timestamp")))
    training = ratings.where("rank <= 0.8").drop("rank")
    test = ratings.where("rank > 0.8").drop("rank")

    training.show()
    test.show()

    avg_utility = training.groupby('movieId').agg(mean('rating').alias("mean_rating")).sort(desc("mean_rating"))
    avg_utility.show()

    pop_baseline = avg_utility.join(test, avg_utility.movieId ==  test.movieId, "inner")
    pop_baseline.show()

    
#    pop_baseline.sort(desc("mean_rating"))
    window = Window.partitionBy(pop_baseline['userId']).orderBy(pop_baseline['mean_rating'].desc())
    pop_baseline.select('*', rank().over(window).alias('rank')).filter(col('rank') <= 100).show(truncate=False)

    

    # Build the recommendation model using ALS on the training data
    # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
              coldStartStrategy="drop")
    model = als.fit(training)

    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))

    # Generate top 10 movie recommendations for each user
    userRecs = model.recommendForAllUsers(10)
    userRecs.show()
    # Generate top 10 user recommendations for each movie
    movieRecs = model.recommendForAllItems(10)
    movieRecs.show()





    # Load the all csv data from ml-latest-small into DataFrame

    #links = spark.read.options(schema = 'movieId INT, imdbId INT, tmdbId INT', header = 'True').csv(f'hdfs:/user/{netID}/movielens/ml-latest-small/links.csv')
    #movies = spark.read.options(schema = 'movieId INT, title STRING, genres STRING', header = 'True').csv(f'hdfs:/user/{netID}/movielens/ml-latest-small//movies.csv')
    #ratings = spark.read.options(schema = 'userId INT, movieId INT, rating FLOAT, timestamp STRING', header = 'True').csv(f'hdfs:/user/{netID}/movielens/ml-latest-small/ratings.csv')
    #ratings = ratings.withColumn("timestamp", from_unixtime(col("timestamp")))    
    #tags = spark.read.options(schema = 'userId INT, movieId INT, tag STRING, timestamp INT', header = 'True').csv(f'hdfs:/user/{netID}/movielens/ml-latest-small/tags.csv')
    #tags = tags.withColumn("timestamp", from_unixtime(col("timestamp")))
    
    #print('Printing links inferred schema')
    #links.printSchema()
    #print('Printing movies inferred schema')
    #movies.printSchema()
    #print('Printing ratings inferred schema')
    #ratings.printSchema()   
    #print('Printing tags inferred schema')
    #tags.printSchema()


    # Give the dataframe a temporary view so we can run SQL queries
    #links.createOrReplaceTempView('links')
    #movies.createOrReplaceTempView('movies')
    #ratings.createOrReplaceTempView('ratings')
    #tags.createOrReplaceTempView('tags')
    
    #tags = tags.sort('timestamp')
    #tags.show()

    # View table format
    #print('links')
    #links = spark.sql('SELECT * FROM links')
    #links.show()

    #print('movies')
    #movies = spark.sql('SELECT * FROM movies')
    #movies.show()

    #print('ratings')
    #ratings = spark.sql('SELECT * FROM ratings')
    #ratings.show()

    #print('tags')
    #tags = spark.sql('SELECT * FROM tags')
    #tags.show()

    #print('number of distinct movies')
    #query1 = spark.sql('SELECT count(distinct movieId) from movies')
    #query1.show()

  #  print('number of distinct users')
  #  query2 = spark.sql('SELECT count(distinct userId) from ratings')
  #  query2.show()

    #query3 = spark.sql('SELECT ratings.userId, ratings.movieId, movies.title, movies.genres from movies, ratings WHERE movies.movieId = ratings.movieId')
    #query3.show()

    #query4 = spark.sql('SELECT userId, count(movieId) FROM ratings GROUP BY userId')
    #query4.show()

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)

