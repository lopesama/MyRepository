# Databricks notebook source
import pyspark.sql.functions as F
from pyspark.sql.types import *
import requests
import json
import re
from datetime import datetime
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# COMMAND ----------

diretorio_part = "dbfs:/databricks-datasets/tpch/data-001/part/"  
diretorio_partsupp = "dbfs:/databricks-datasets/tpch/data-001/partsupp/"

#Retrieving data from the file part
df_part = spark.read.csv(diretorio_part,inferSchema=True ,sep="|")

#Retrieving data from the file partsupp
df_partsupp = spark.read.csv(diretorio_partsupp,inferSchema=True, sep="|")

# COMMAND ----------

# MAGIC %md
# MAGIC Section 1 = > TPC-H Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Question #1: Joins in Core Spark Pick any two datasets and join them using Spark's API. Feel free to pick any two datasets. For example: PART and PARTSUPP. The goal of this exercise is not to derive anything meaningful out of this data but to demonstrate how to use Spark to join two datasets. For this problem you're NOT allowed to use SparkSQL. You can only use RDD API. You can use either Python or Scala to solve this problem.

# COMMAND ----------

# Converting the Spark DataFrames to RDD
rdd_part = df_part.rdd
rdd_partsupp = df_partsupp.rdd

# Creating keys for join
rdd_part_keyed = rdd_part.keyBy(lambda row: row['_c0'])
rdd_partsupp_keyed = rdd_partsupp.keyBy(lambda row: row['_c0'])

# Join
rdd_joined = rdd_part_keyed.join(rdd_partsupp_keyed)

# Selecting the required fields
rdd_final = rdd_joined.map(lambda x: (
    x[0],
    x[1][1]['_c1'],
    x[1][0]['_c2'],
    x[1][0]['_c3'],
    x[1][0]['_c4'],
))

rdd_final.collect()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Question #2: Joins With Spark SQL
# MAGIC Pick any two datasets and join them using SparkSQL API. Feel free to pick any two datasets. For example: PART and PARTSUPP. The goal of this exercise is not to derive anything meaningful out of this data but to demonstrate how to use Spark to join two datasets. For this problem you're NOT allowed to use the RDD API. You can only use SparkSQL API. You can use either Python or Scala to solve this problem.

# COMMAND ----------

#Performing the join between the two DataFrames with Spark

df_final = (
    df_part.alias('part')
        .join(df_partsupp.alias('supp')
              , [F.col('part._c0') == F.col('supp._c0')]
              , how='inner')
        .select(
            F.col('part._c0')
            , F.col('supp._c1')
            , F.col('part._c2')
            , F.col('part._c3')
            , F.col('part._c4')
        )

)

df_final.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC Question #3: Alternate Data Formats
# MAGIC The given dataset above is in raw text storage format. What other data storage format can you suggest to optimize the performance of our Spark workload if we were to frequently scan and read this dataset. Please come up with a code example and explain why you decide to go with this approach. Please note that there's no completely correct answer here. We're interested to hear your thoughts and see the implementation details.shell/1282

# COMMAND ----------

# First, I would save the files in Parquet format because of its compatibility, performance, benefits, and support for various data types with the following code.

path_part = "/Users/victorpereirasilva826@gmail.com/part"  
path_partsupp = "/Users/victorpereirasilva826@gmail.com/partsupp"

df_part.write.parquet(path_part, mode='overwrite')
df_partsupp.write.parquet(path_partsupp, mode='overwrite')

# Secondly, I would read the saved Parquet file

df_part_parquet = spark.read.parquet(path_part)
df_partsupp = spark.read.parquet(path_partsupp)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Section 2 = > Baby Names Datase
# MAGIC
# MAGIC This dataset comes from a website referenced by Data.gov. It lists baby names used in the state of NY from 2007 to 2012. Use this JSON file as an input and answer the 3 questions accordingly.
# MAGIC
# MAGIC https://health.data.ny.gov/api/views/jxy9-yhdk/rows.json

# COMMAND ----------

display(dbutils.fs.ls("./"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Question #1: Spark SQL's Native JSON Support
# MAGIC Use Spark SQL's native JSON support to create a temp table you can use to query the data (you'll use the registerTempTable operation). Show a simple sample query.

# COMMAND ----------

api_url = "https://health.data.ny.gov/api/views/jxy9-yhdk/rows.json"
response = requests.get(api_url)

dados_json = response.json()
df = spark.createDataFrame([dados_json])

output_path = "./api_dados.json" 
df.write.json(output_path, mode='overwrite') 

# COMMAND ----------

# Read the JSON file
json_file_path = "dbfs:/api_dados.json/" 
df = spark.read.json(json_file_path)

# Create a replace temp view
df.createOrReplaceTempView("json_table") 

result_df = spark.sql("SELECT * FROM json_table")

result_df.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC Question #2: Working with Nested Data
# MAGIC What does the nested schema of this dataset look like? How can you bring these nested fields up to the top level in a DataFrame?

# COMMAND ----------


def flatten_df(nested_df):
    while True:
        nested_columns = [col for col in nested_df.schema.fields if isinstance(col.dataType, (StructType, ArrayType))]
        if not nested_columns:
            break
        
        flat_columns = []
        for field in nested_df.schema.fields:
            if isinstance(field.dataType, StructType):
                for nested_field in field.dataType.fields:
                    flat_columns.append(
                        F.col(f"{field.name}.{nested_field.name}").alias(f"{field.name}_{nested_field.name}")
                    )
            elif isinstance(field.dataType, ArrayType):
                flat_columns.append(F.explode(f"{field.name}").alias(f"{field.name}_exploded"))
            else:
                flat_columns.append(col(field.name))
        
        nested_df = nested_df.select(*flat_columns)

    return nested_df
    
df = spark.read.json("dbfs:/api_dados.json/")
flattened_df = flatten_df(df)

flattened_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Question #3: Executing Full Data Pipelines
# MAGIC Create a second version of the answer to Question 2, and make sure one of your queries makes the original web call every time a query is run, while another version only executes the web call one time.

# COMMAND ----------

# MAGIC %md
# MAGIC Question #4: Analyzing the Data
# MAGIC Using the tables you created, create a simple visualization that shows what is the most popular first letters baby names to start with in each year.

# COMMAND ----------

# MAGIC %md
# MAGIC Section 3 => Log Processing
# MAGIC The following data comes from the Learning Spark book.

# COMMAND ----------

display(dbutils.fs.ls("/databricks-datasets/learning-spark/data-001/fake_logs"))

# COMMAND ----------

display(dbutils.fs.head("/databricks-datasets/learning-spark/data-001/fake_logs/log1.log"))

# COMMAND ----------

# MAGIC %md
# MAGIC Question #1: Parsing Logs
# MAGIC Parse the logs in to a DataFrame/Spark SQL table that can be queried. This should be done using the Dataset API.

# COMMAND ----------

# Defining the schema
log_schema = StructType([
    StructField("ip_address", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("http_method", StringType(), True),
    StructField("url", StringType(), True),
    StructField("response_code", IntegerType(), True),
])

# Regular expression to parse the logs
log_pattern = r'^(\S+) \S+ \S+ \[(.*?)\] "(.*?) (.*?) HTTP/1.\d" (\d+)'

# Function to parse a log line
def parse_log_line(log_line):
    match = re.match(log_pattern, log_line)
    if match:
        return {
            "ip_address": match.group(1),
            "timestamp": match.group(2),
            "http_method": match.group(3),
            "url": match.group(4),
            "response_code": int(match.group(5)),
        }
    return None

# Read the log file as a text file
log_rdd = spark.sparkContext.textFile("/databricks-datasets/learning-spark/data-001/fake_logs/log1.log")

# Parse the logs using the defined schema
parsed_rdd = log_rdd.map(parse_log_line).filter(lambda x: x is not None)

# Create a DataFrame from the parsed data
log_df = spark.createDataFrame(parsed_rdd, schema=log_schema)

# Register the DataFrame as a temporary Spark SQL table
log_df.createOrReplaceTempView("logs")

# Execute a Spark SQL query
result = spark.sql("SELECT * FROM logs")

# Display the results
result.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC Question #2: Analysis
# MAGIC Generate some insights from the log data.

# COMMAND ----------

# In the log, we can see the request IP, the date and time of the request, and the response obtained

# COMMAND ----------

# MAGIC %md
# MAGIC Section 4 => CSV Parsing
# MAGIC The following examples involove working with simple CSV data

# COMMAND ----------

# MAGIC %md
# MAGIC val full_csv = sc.parallelize(Array(
# MAGIC   "col_1, col_2, col_3",
# MAGIC   "1, ABC, Foo1",
# MAGIC   "2, ABCD, Foo2",
# MAGIC   "3, ABCDE, Foo3",
# MAGIC   "4, ABCDEF, Foo4",
# MAGIC   "5, DEF, Foo5",
# MAGIC   "6, DEFGHI, Foo6",
# MAGIC   "7, GHI, Foo7",
# MAGIC   "8, GHIJKL, Foo8",
# MAGIC   "9, JKLMNO, Foo9",
# MAGIC   "10, MNO, Foo10"))
# MAGIC
# MAGIC Question #1: CSV Header Rows
# MAGIC Given the simple RDD full_csv below, write the most efficient Spark job you can to remove the header row

# COMMAND ----------

# Creating the dataframe
full_csv = sc.parallelize([
    "col_1, col_2, col_3",
    "1, ABC, Foo1",
    "2, ABCD, Foo2",
    "3, ABCDE, Foo3",
    "4, ABCDEF, Foo4",
    "5, DEF, Foo5",
    "6, DEFGHI, Foo6",
    "7, GHI, Foo7",
    "8, GHIJKL, Foo8",
    "9, JKLMNO, Foo9",
    "10, MNO, Foo10"
])

# Filtering the row with the header
header = full_csv.first().split(",")

# extracting information
data_rdd = full_csv.zipWithIndex().filter(lambda x: x[1] > 0).keys()

# Creating a dataframe with the result
df = data_rdd.map(lambda row: Row(*row.split(","))).toDF(header)

# Removing the header
without_header = full_csv.zipWithIndex().filter(lambda x: x[1] > 0).keys()

# Display
without_header.collect()

# COMMAND ----------

# MAGIC %md
# MAGIC Question #2: SparkSQL Dataframes
# MAGIC Using the full_csv RDD above, write code that results in a DataFrame where the schema was created programmatically based on the heard row. Create a second RDD similair to full_csv and uses the same function(s) you created in this step to make a Dataframe for it.

# COMMAND ----------

# Creating the dataframe

full_csv = sc.parallelize([
    "col_1, col_2, col_3",
    "1, ABC, Foo1",
    "2, ABCD, Foo2",
    "3, ABCDE, Foo3",
    "4, ABCDEF, Foo4",
    "5, DEF, Foo5",
    "6, DEFGHI, Foo6",
    "7, GHI, Foo7",
    "8, GHIJKL, Foo8",
    "9, JKLMNO, Foo9",
    "10, MNO, Foo10"
])

# Filtering the row with the header
header = full_csv.first().split(", ")

# Building the schema
schema = StructType([StructField(col, StringType(), True) for col in header])

# Creating the function to remove the header line
def create_df_from_rdd(rdd):
    rdd_data = rdd.zipWithIndex().filter(lambda x: x[1] > 0).keys()
    df = rdd_data.map(lambda row: Row(*row.split(", "))).toDF(schema)
    return df

# Creating the dataframe
df1 = create_df_from_rdd(full_csv)

df1.show()

# COMMAND ----------

# Creating the dataframe
full_csv_2 = sc.parallelize([
    "col_1, col_2, col_3",
    "11, XYZ, Bar1",
    "12, XYZA, Bar2",
    "13, XYZAB, Bar3",
    "14, XYZABC, Bar4",
    "15, XYZABCD, Bar5"
])

# Creating the dataframe
df2 = create_df_from_rdd(full_csv_2)

df2.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Question #3: Parsing Pairs
# MAGIC Write a Spark job that processes comma-seperated lines that look like the below example to pull out Key Value pairs.
# MAGIC
# MAGIC Given the following data:
# MAGIC
# MAGIC Row-Key-001, K1, 10, A2, 20, K3, 30, B4, 42, K5, 19, C20, 20
# MAGIC Row-Key-002, X1, 20, Y6, 10, Z15, 35, X16, 42
# MAGIC Row-Key-003, L4, 30, M10, 5, N12, 38, O14, 41, P13, 8
# MAGIC You'll want to create an RDD that contains the following data:
# MAGIC
# MAGIC Row-Key-001, K1
# MAGIC Row-Key-001, A2
# MAGIC Row-Key-001, K3
# MAGIC Row-Key-001, B4
# MAGIC Row-Key-001, K5
# MAGIC Row-Key-001, C20
# MAGIC Row-Key-002, X1
# MAGIC Row-Key-002, Y6
# MAGIC Row-Key-002, Z15
# MAGIC Row-Key-002, X16
# MAGIC Row-Key-003, L4
# MAGIC Row-Key-003, M10
# MAGIC Row-Key-003, N12
# MAGIC Row-Key-003, O14
# MAGIC Row-Key-003, P13

# COMMAND ----------

# Creating the data
data = [
    "Row-Key-001, K1, 10, A2, 20, K3, 30, B4, 42, K5, 19, C20, 20",
    "Row-Key-002, X1, 20, Y6, 10, Z15, 35, X16, 42",
    "Row-Key-003, L4, 30, M10, 5, N12, 38, O14, 41, P13, 8"
]

# Building a RDD
rdd = spark.sparkContext.parallelize(data)

# Defining the function
def extract_key_value_pairs(row):
    elements = row.split(", ")
    primary_key = elements[0]
    secondary_keys = [elements[i] for i in range(1, len(elements), 2)]
    return [(primary_key, sk) for sk in secondary_keys]

# Saving the result
result_rdd = rdd.flatMap(extract_key_value_pairs)

result_rdd.collect()

# COMMAND ----------

# MAGIC %md
# MAGIC Section 5 => Connecting to JDBC Database
# MAGIC Write a Spark job that queries MySQL using its JDBC Driver.
# MAGIC
# MAGIC Load your JDBC Driver onto Databricks
# MAGIC Databricks comes preloaded with JDBC libraries for mysql, but you can attach other JDBC libraries and reference them in your code
# MAGIC See our Libraries Notebook for instructions on how to install a Java JAR.

# COMMAND ----------

jdbc_url = "url to mysql database"
jdbc_driver = "com.mysql.cj.jdbc.Driver"
jdbc_user = "user"
jdbc_password = "password to enter the server"
table_name = "name of the table"

df = spark.read.format("jdbc").options(
    url=jdbc_url,
    driver=jdbc_driver,
    dbtable=table_name,
    user=jdbc_user,
    password=jdbc_password
).load()

df.show()

#I didn't quite understand the request, but above there's an example of how to connect with a MySQL JDBC. Since I don't have a username, password, server, and table, I left them with default values

# COMMAND ----------

# MAGIC %md
# MAGIC Section 6 => Create Tables Programmatically And Cache The Table
# MAGIC Create a table using Scala or Python
# MAGIC
# MAGIC Use CREATE EXTERNAL TABLE in SQL, or DataFrame.saveAsTable() in Scala or Python, to register tables.
# MAGIC Please refer to the Accessing Data guide for how to import specific data types.
# MAGIC Temporary Tables
# MAGIC Within each Spark cluster, temporary tables registered in the sqlContext with DataFrame.registerTempTable will also be shared across the notebooks attached to that Databricks cluster.
# MAGIC Run someDataFrame.registerTempTable(TEMP_TABLE_NAME) to give register a table.
# MAGIC These tables will not be visible in the left-hand menu, but can be accessed by name in SQL and DataFrames.

# COMMAND ----------

data = [("Victor", 27), ("Leticia", 27), ("Junior", 33)]
columns = ["name", "age"]
df = spark.createDataFrame(data, columns)

df.write.mode("overwrite").saveAsTable("family_table")

spark.sql("SELECT * FROM family_table").show()

# COMMAND ----------

# MAGIC %md
# MAGIC Section 7 => DataFrame UDFs and DataFrame SparkSQL Functions
# MAGIC Below we've created a small DataFrame. You should use DataFrame API functions and UDFs to accomplish two tasks.
# MAGIC
# MAGIC You need to parse the State and city into two different columns.
# MAGIC You need to get the number of days in between the start and end dates. You need to do this two ways.
# MAGIC Firstly, you should use SparkSQL functions to get this date difference.
# MAGIC Secondly, you should write a udf that gets the number of days between the end date and the start date.
# MAGIC

# COMMAND ----------


from pyspark.sql import functions as F
from pyspark.sql.types import *

# Build an example DataFrame dataset to work with. 
dbutils.fs.rm("/tmp/dataframe_sample.csv", True)
dbutils.fs.put("/tmp/dataframe_sample.csv", """id|end_date|start_date|location
1|2015-10-14 00:00:00|2015-09-14 00:00:00|CA-SF
2|2015-10-15 01:00:20|2015-08-14 00:00:00|CA-SD
3|2015-10-16 02:30:00|2015-01-14 00:00:00|NY-NY
4|2015-10-17 03:00:20|2015-02-14 00:00:00|NY-NY
5|2015-10-18 04:30:00|2014-04-14 00:00:00|CA-SD
""", True)

formatPackage = "csv" if sc.version > '1.6' else "com.databricks.spark.csv"
df = sqlContext.read.format(formatPackage).options(header='true', delimiter = '|').load("/tmp/dataframe_sample.csv")

df.printSchema()

# COMMAND ----------

# Creating the column state and city
df_with_location = df.withColumn(
    "state",
    F.split(F.col("location"), "-").getItem(0)
).withColumn(
    "city",
    F.split(F.col("location"), "-").getItem(1)
)

# Creating the column days_difference with datediff from spark
df_with_days_difference = df_with_location.withColumn(
    "days_difference",
    F.datediff(F.col("end_date"), F.col("start_date"))
)

df_with_days_difference.show()

# COMMAND ----------

# Defining a function to calculate the days between
def date_diff_udf(end_date_str, start_date_str):
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d %H:%M:%S")
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")
    return (end_date - start_date).days

udf_date_diff = F.udf(date_diff_udf, IntegerType())

# Creating a new dataframe with the UDF function
df_with_udf_days_difference = df_with_location.withColumn(
    "udf_days_difference",
    udf_date_diff(F.col("end_date"), F.col("start_date"))
)

df_with_udf_days_difference.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC Section 8 => Machine Learning

# COMMAND ----------

# MAGIC %md
# MAGIC Question 1: Demonstrate The Use of a MLlib Algorithm Using the DataFrame Interface(org.apache.spark.ml).
# MAGIC Demonstrate use of an MLlib algorithm and show an example of tuning the algorithm to improve prediction accuracy.
# MAGIC
# MAGIC Use Decision Tree Example using Databricks MLib.

# COMMAND ----------

data = [
    (1, Vectors.dense(1.0, 0.5, 1.0), 1.0),
    (2, Vectors.dense(2.0, 1.5, 1.1), 0.0),
    (3, Vectors.dense(3.0, 1.0, 1.2), 1.0),
    (4, Vectors.dense(4.0, 1.5, 1.3), 0.0),
    (5, Vectors.dense(5.0, 1.7, 1.4), 1.0),
]
columns = ["id", "features", "label"]

df = spark.createDataFrame(data, columns)

# COMMAND ----------

dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

model = dt.fit(df)

print(model.toDebugString)

# COMMAND ----------

test_data = [
    (6, Vectors.dense(1.1, 0.4, 1.1), 1.0),
    (7, Vectors.dense(2.1, 1.6, 1.2), 0.0),
    (8, Vectors.dense(3.1, 1.2, 1.2), 1.0),
    (9, Vectors.dense(4.2, 1.6, 1.3), 0.0),
    (10, Vectors.dense(5.1, 1.8, 1.5), 1.0),
]

test_df = spark.createDataFrame(test_data, columns)

predictions = model.transform(test_df)

predictions.show()

evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = ", accuracy)

# COMMAND ----------

param_grid = ParamGridBuilder() \
    .addGrid(dt.maxDepth, [2, 4, 6]) \
    .addGrid(dt.maxBins, [10, 20, 40]) \
    .build()

cross_val = CrossValidator(
    estimator=dt,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3
)

cv_model = cross_val.fit(df)

cv_predictions = cv_model.transform(test_df)

cv_accuracy = evaluator.evaluate(cv_predictions)
print("Cross-validated test set accuracy = ", cv_accuracy)

# COMMAND ----------

# MAGIC %md 
# MAGIC Section 9 => XML, Sql understanding, Streaming, Debugging Spark

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Question 1 : Create sample large xml data in your own with namespace and generate dataframe and do some DataSet transformations and actions. Also provide some meaningful insights from it.

# COMMAND ----------

# MAGIC %md
# MAGIC Question 2 : You have the SQL query below to find employees who earn the top three salaries in each of the department. Please write a pure Java/Scala code equivalent to this SQL code. Do not use direct SQL inside the java/scala code. But we need you to use your programming skills to answer the question
# MAGIC
# MAGIC %sql 
# MAGIC WITH order_salary AS
# MAGIC ( 
# MAGIC SELECT dept.Name AS DeptName, emp.Name AS EmpName,emp.Salary AS Salary, RANK() OVER (PARTITION BY dept.Id ORDER BY emp.Salary DESC) OrderedRank
# MAGIC FROM Employee AS emp
# MAGIC INNER JOIN Department AS dept on dept.Id = emp.DepartmentId 
# MAGIC )
# MAGIC SELECT DeptName, EmpName, Salary FROM order_salary WHERE OrderedRank <= 3;

# COMMAND ----------

# MAGIC %scala
# MAGIC
# MAGIC import org.apache.spark.sql.{SparkSession, DataFrame}
# MAGIC import org.apache.spark.sql.functions._
# MAGIC import org.apache.spark.sql.expressions.Window
# MAGIC
# MAGIC val employeeData = Seq(
# MAGIC   (1, "John", 50000, 1),
# MAGIC   (2, "Jane", 60000, 1),
# MAGIC   (3, "Jim", 70000, 2),
# MAGIC   (4, "Jack", 80000, 2),
# MAGIC   (5, "Jill", 90000, 2),
# MAGIC   (6, "Joe", 100000, 1),
# MAGIC   (7, "Jake", 110000, 3),
# MAGIC   (8, "Jessie", 120000, 3),
# MAGIC   (9, "Jeremy", 130000, 3),
# MAGIC   (10, "Judy", 140000, 3)
# MAGIC )
# MAGIC
# MAGIC val departmentData = Seq(
# MAGIC   (1, "HR"),
# MAGIC   (2, "Engineering"),
# MAGIC   (3, "Marketing")
# MAGIC )
# MAGIC
# MAGIC val employeeDF = spark.createDataFrame(employeeData).toDF("Id", "Name", "Salary", "DepartmentId")
# MAGIC val departmentDF = spark.createDataFrame(departmentData).toDF("Id", "DepartmentName")
# MAGIC
# MAGIC val joinedDF = employeeDF.join(departmentDF, employeeDF("DepartmentId") === departmentDF("Id"))
# MAGIC
# MAGIC val windowSpec = Window.partitionBy("Name").orderBy(desc("Salary"))
# MAGIC
# MAGIC val rankedDF = joinedDF.withColumn("Rank", rank().over(windowSpec))
# MAGIC
# MAGIC val topThreeSalaries = rankedDF.filter($"Rank" <= 3)
# MAGIC
# MAGIC val resultDF = topThreeSalaries.select("Name", "DepartmentName", "Salary")
# MAGIC
# MAGIC resultDF.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Question 3 : There is a customer who reports that one of his spark job stalls at the last executor (399/400) for more than 1hr and executor logs are showing ‘ExecutorLost’ stacktrace and couple of retry attempts. What do you recommend as a next step(s)?

# COMMAND ----------

# First we have to examine the executor logs to understand the cause of the error.

# After that we need to verify if the resources available for spark is sufficient.

# We can also revie the configuration setting and restart the affected executor.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Question 4 : Consider the following streaming job What does the below code do? How long will it retain the state? And why?
# MAGIC
# MAGIC val streamingDf = readStream() streamingDf.withWatermark("eventTime","10 seconds").dropDuplicates("guid") streamingDf.writeStream()

# COMMAND ----------

# This fuction is reading a data from a streaming source, applying a watermark to track of how much data can be considered out-of-date, in this case with the value of 10 seconds, spark will conider records with timestamps older than 10 seconds as late. 
# 
# After that, with the dropDuplicates, the function are removing the duplicate records on the column named "guid". 
# 
# The writeStream parameter defines where the output of the straming will be written.
#
#With the watermark of 10 seconds, spark retains information related to the eventTime column for at least 10 seconds. 
# 
# Is necessary to maintain information about which records have already been processed
