# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 06:33:07 2018

@author: Kumar
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import findspark
findspark.init("C:\\Users\\Kumar\\spark-2.3.1-bin-hadoop2.7")
from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .master("local") \
    .appName("chicagotaxi1") \
    .getOrCreate()
dataset = spark.read.csv('C:\\Users\\Kumar\\.spyder-py3\\chicago-taxi-rides-2016 (1)\\*', header = True, inferSchema = True)
dataset.printSchema()
dataset.show(5)
dataset= dataset.select('trip_seconds','trip_miles','fare')
dataset.printSchema()
dataset = dataset.na.drop()
dataset = dataset.filter((dataset.trip_seconds> 0) & (dataset.trip_miles>0))

assembler = VectorAssembler(inputCols=['trip_seconds','trip_miles','fare'], outputCol='features')
assembled_data = assembler.transform(dataset)
assembled_data.show(5)
scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures')

scaler_model = scaler.fit(assembled_data)
scaled_data = scaler_model.transform(assembled_data)
scaled_data.printSchema()
scaled_data.show(5)

kmeans = KMeans().setK(2).setSeed(1)
model = kmeans.fit(scaled_data)
predictions = model.transform(scaled_data)

predictions.groupBy('prediction').count().show()
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)
import numpy as np
evaluator = ClusteringEvaluator()
cost = np.zeros(20) 
silhouette = np.zeros(20) 
for k in range(2,20):

    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
    model = kmeans.fit(scaled_data.sample(False,0.1, seed=42))
    cost[k] = model.computeCost(scaled_data) 
    predictions = model.transform(scaled_data)
    silhouette[k] = evaluator.evaluate(predictions)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,20),cost[2:20])
ax.set_xlabel('k')
ax.set_ylabel('cost')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,20),silhouette[2:20])
ax.set_xlabel('k')
ax.set_ylabel('silhouette')
    

