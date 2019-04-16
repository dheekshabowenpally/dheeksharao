# -*- coding: utf-8 -*-ll
"""
Created on Tue Dec  4 16:33:13 2018

@author: genre
"""

import findspark
findspark.init("C:\\Users\\Kumar\\spark-2.3.1-bin-hadoop2.7")
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

spark = SparkSession\
        .builder\
        .appName("FinalProjectWork")\
        .getOrCreate()
# reading the data into DataFrame        
df = spark.read.load("C:\\Users\\Kumar\\.spyder-py3\\chicago-taxi-rides-2016 (1)\\chicago_taxi_trips_2016_01.csv", 
                     format = "csv",sep= ",", inferSchema= "true", header= "true")

#printing the data structure
df.printSchema() 
df.show(5) 
df = df.select(['trip_seconds', 'trip_miles','fare'])
# drop the null values (null value Rows)
df = df.na.drop()
df = df.filter((df.trip_seconds> 0) & (df.trip_miles>0))
df.show(5) 
# Features 
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import *
vectorAssembler = VectorAssembler(inputCols = ['trip_seconds', 'trip_miles' ], outputCol = 'features')
Taxi_df = vectorAssembler.transform(df)
Taxi_df = Taxi_df.select(['trip_seconds', 'trip_miles','features',col('fare').alias("label")])
Taxi_df.count()
Taxi_df.show()


from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder

train, test = Taxi_df.randomSplit([0.70,0.30], seed=2018)  # devided the dataset into 70, 30 percentage training and test res.
train.show()
test.show()
lr= LinearRegression()
#lr=LinearRegression(maxIter=10) # checking with 10 iterations to check how much time it will take 

# paramgrid building
paramGrid = ParamGridBuilder()\
        .addGrid(lr.regParam,[0.01, 0.05, 0.1,0.5, 1.0])\
        .addGrid(lr.fitIntercept, [False, True])\
        .addGrid(lr.elasticNetParam,[0.0, 0.25,  0.5, 0.75, 1.0])\
        .build()
#print("LinearRegression parameters:\n" + lr.explainParams() + "\n")  

TrainVal = TrainValidationSplit(estimator=lr,
                              estimatorParamMaps=paramGrid,
                              evaluator=RegressionEvaluator(),
                              trainRatio=0.8) 
# Run TrainValidationSplit, and choose the best set of parameters.
tvmodel = TrainVal.fit(train)

#print(paramGrid)  
best_model = tvmodel.bestModel
print( 'Best Param (regParam): = {0}'.format( best_model._java_obj.getRegParam()))
print( 'Best Param (Intercept): = {0}'.format( best_model._java_obj.getFitIntercept()))
print('Best Param (elasticNetParam): ={0}'.format(best_model._java_obj.getElasticNetParam()))
tvmodel.transform(test).show() 


###

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol='label', maxIter=10, regParam=0.01, elasticNetParam=0.0)
lr_model = lr.fit(train)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))
trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

lr_predictions = lr_model.transform(test)
lr_predictions.select("prediction","label","features").show(5)
from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="label",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))
test_result = lr_model.evaluate(test)
print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)
predictions = lr_model.transform(test)
predictions.select("prediction","label","features").show()




