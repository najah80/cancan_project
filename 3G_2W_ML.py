from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('3G_2week_ml').getOrCreate()

#Import libraries
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer, IndexToString
from pyspark.ml.feature import VectorAssembler, PCA, StandardScaler
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, OneVsRest, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

import pandas as pd
import numpy as np
pd.set_option('display.float_format', lambda x: '%.2f' % x)
from itertools import chain

#read result of two weeks aggregation 
df = spark.read.parquet("/user/ngwh3132/WORK/cancan_project/3G_2W/new_df_3G_2weeks.parquet")

# Create flags for columns containing NaN values and then, replace all NaN with 'O'
df = df.withColumn('rg_missing', (F.col('rg').isNull()).cast('int')) \
       .withColumn('ie_avg_missing', (F.col('ie_avg').isNull()).cast('int')) \
       .withColumn('ie_entropy_missing', (F.col('ie_entropy').isNull()).cast('int')) \
       .withColumn('pos_entropy_missing', (F.col('pos_entropy').isNull()).cast('int')) \
       .withColumn('percent_nocturnal_missing', (F.col('percent_nocturnal').isNull()).cast('int')) \
       .withColumn('nb_event_day_missing', (F.col('nb_event_day').isNull()).cast('int')) \
       .withColumn('nb_event_night_missing', (F.col('nb_event_night').isNull()).cast('int')) \

df_new = df.fillna(0, subset=['nb_event_day', 'nb_event_night', 'ie_avg', 'ie_entropy', 'pos_entropy', 'rg', 'percent_nocturnal'])

#Separate class null from the rest of data and save it to use it after building ML model to make prediction on
df_null = df_new.filter(F.col('class') == 'null')
# df_null.coalesce(1).write.parquet("/user/ngwh3132/WORK/cancan_project/3G_2W/df_null_3G_2weeks.parquet", mode='overwrite')

#let's take the labeled data 
df_new = df_new.filter(F.col('class') != 'null').withColumn("idx", F.monotonically_increasing_id())

#take only numerical features that are interesting to build the model
features = ['nb_events','nb_event_day','nb_event_night','ie_avg','ie_entropy','pos_entropy','rg','nb_call','nb_sms','nb_data','percent_nocturnal','rg_missing','ie_avg_missing','ie_entropy_missing','pos_entropy_missing','percent_nocturnal_missing', 'nb_event_day_missing', 'nb_event_night_missing']

#String Indexer
indexer = StringIndexer(inputCol="class", outputCol="label")
df_new = indexer.fit(df_new).transform(df_new).withColumn("label", F.col('label').cast('int'))

#Give Weights to Instances:
#Since we have hight imbalanced classes and pyspark is not yet supporting class weights feature, I will do it manually. Here’s how we can #compute “balanced” weights with data from a PySpark DataFrame.
y_collect = df_new.select("label").groupBy("label").count().collect()
unique_y = [x["label"] for x in y_collect]
total_y = sum([x["count"] for x in y_collect])
unique_y_count = len(y_collect)
bin_count = [x["count"] for x in y_collect]

class_weights_spark = {i: ii for i, ii in zip(unique_y, total_y / (unique_y_count * np.array(bin_count)))}

#PySpark needs to have a weight assigned to each instance (i.e., row) in the training set. I create a mapping to apply a weight to each #training instance.
mapping_expr = F.create_map([F.lit(x) for x in chain(*class_weights_spark.items())])
df_weighted = df_new.withColumn("weight", mapping_expr.getItem(F.col("label")))

features_2 = ['nb_events','nb_event_day','nb_event_night','ie_avg','ie_entropy','pos_entropy','rg','nb_call','nb_sms','nb_data','percent_nocturnal','rg_missing','ie_avg_missing','ie_entropy_missing','pos_entropy_missing','percent_nocturnal_missing', 'nb_event_day_missing', 'nb_event_night_missing']

df_features = df_weighted.select(features_2)

#Vector Assembler:
#The vector assembler is basically use to concatenate all the features into a single vector which can be further passed to the estimator #or ML algorithm. I assemble all the input features into a vector.
assembler = VectorAssembler(inputCols=df_features.columns, outputCol="features")
df_new = assembler.transform(df_weighted)

#StandardScaler
standardscaler=StandardScaler().setInputCol("features").setOutputCol("Scaled_features")
df_new = standardscaler.fit(df_new).transform(df_new)

#Split dataset to train and test sets
(trainingData, testData) = df_new.select('Scaled_features', 'label', 'weight').randomSplit([0.7, 0.3], seed=1234)

#Logistic Regession
#Without the instance weights, the model predicts all instances as the frequent class
lr = LogisticRegression(featuresCol="Scaled_features", labelCol="label", weightCol="weight", family = 'multinomial')
lrModel = lr.fit(trainingData)

# With the weights, the model assigns the same number of instances to each class (even the less commmon one).
lrModel.transform(trainingData).agg(F.mean("prediction")).show()

predictions = lrModel.transform(testData)

#Model evaluation

#f1 metric
evaluator = MulticlassClassificationEvaluator(\
labelCol="label", predictionCol="prediction",\
metricName="f1")
f1_score = evaluator.evaluate(predictions)
print("Test f1 score = %g " % (f1_score))

#weightedPrecision metric
evaluator = MulticlassClassificationEvaluator(\
labelCol="label", predictionCol="prediction",\
metricName="weightedPrecision")
f1_score = evaluator.evaluate(predictions)
print("Test weightedPrecision = %g " % (f1_score))

#weightedRecall metric
evaluator = MulticlassClassificationEvaluator(\
labelCol="label", predictionCol="prediction",\
metricName="weightedRecall")
f1_score = evaluator.evaluate(predictions)
print("Test weightedRecall = %g " % (f1_score))

#accuracy metric
evaluator = MulticlassClassificationEvaluator(\
labelCol="label", predictionCol="prediction",\
metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test accuracy = %g " % (accuracy))

#Print the coefficients and intercept for multinomial logistic regression
print("Coefficients: \n" + str(lrModel.coefficientMatrix))
print("Intercept: " + str(lrModel.interceptVector))

#Model Tuning and CrossValidator
#build paramgrid with hyperparamter for tuning
paramGrid = ParamGridBuilder()\
 .addGrid(lr.aggregationDepth,[2,5,10])\
 .addGrid(lr.elasticNetParam,[0.0, 0.5, 1.0])\
 .addGrid(lr.fitIntercept,[False, True])\
 .addGrid(lr.maxIter,[10, 100, 1000])\
 .addGrid(lr.regParam,[0.01, 0.5, 2.0]) \
 .build()

#Then we create 2-fold Cross-validator
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=MulticlassClassificationEvaluator(\
labelCol="label", predictionCol="prediction",\
metricName="f1"), numFolds=2)

# Run cross validations
cvModel = cv.fit(trainingData)
predict_test=cvModel.transform(testData)
print("After hyperparameter tuning, f1 score for test set is {}".format(evaluator.evaluate(predict_test)))

#End
