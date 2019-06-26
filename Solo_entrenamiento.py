# Databricks notebook source
# MAGIC %md
# MAGIC ### Cargamos la las bibliotecas necesarias

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import lit, count, sum, when, from_json, col
from datetime import datetime
from datetime import timedelta
import requests
from pyspark.sql.types import StructType, StructField, FloatType, StringType, IntegerType, LongType, DoubleType

# COMMAND ----------

# MAGIC %md
# MAGIC ### Declaramos el esquema de las columnas que utilizaremos para el entrenamiento y evaluación

# COMMAND ----------

esquema_de_streaming = StructType([
  StructField( 'funded_amnt', FloatType(), True ),
  StructField( 'int_rate', FloatType(), True ),
  StructField( 'installment', FloatType(), True ),
  StructField( 'annual_inc', FloatType(), True ),
  StructField( 'dti', FloatType(), True ),
  StructField( 'delinq_2yrs', FloatType(), True ),
  StructField( 'term', StringType(), True ),
  StructField( 'emp_length', StringType(), True ),
  StructField( 'home_ownership', StringType(), True ),
  StructField( 'verification_status', StringType(), True ),
  StructField( 'purpose', StringType(), True ),
  StructField( 'clasificacion_de_prestamo', IntegerType(), True ),
])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cargamos la información con la entrenaremos el modelo

# COMMAND ----------

conjunto_entrenamiento = (spark.read
  .option("HEADER", True)
  .option( 'delimiter', ',' )
  .option("inferSchema", True)
  .csv("/mnt/cfdis_sat/Demo AXA/set_de_prueba.csv")
  .select( 'funded_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'delinq_2yrs', 'term', 'emp_length', 'home_ownership', 'verification_status', 'purpose', 'clasificacion_de_prestamo' )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Llenamos los valores nulos con 0, si son números, o '', si son cadenas.

# COMMAND ----------

conjunto_entrenamiento = (conjunto_entrenamiento
                          .na.fill(0, [ 'funded_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'delinq_2yrs' ])
                          .na.fill('', [ 'term', 'emp_length', 'home_ownership', 'verification_status', 'purpose'  ])
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Declaramos y entrenamos los indexadores que utilizamos para convertir las columnas de cadena a número

# COMMAND ----------

indexer_term = StringIndexer(inputCol="term", outputCol="term2").fit(conjunto_entrenamiento)
conjunto_entrenamiento = indexer_term.transform(conjunto_entrenamiento)

# COMMAND ----------

indexer_emp_length = StringIndexer(inputCol="emp_length", outputCol="emp_length2").fit(conjunto_entrenamiento)
conjunto_entrenamiento = indexer_emp_length.transform(conjunto_entrenamiento)

# COMMAND ----------

indexer_home_ownership = StringIndexer(inputCol="home_ownership", outputCol="home_ownership2").fit(conjunto_entrenamiento)
conjunto_entrenamiento = indexer_home_ownership.transform(conjunto_entrenamiento)

# COMMAND ----------

indexer_verification_status = StringIndexer(inputCol="verification_status", outputCol="verification_status2").fit(conjunto_entrenamiento)
conjunto_entrenamiento = indexer_verification_status.transform(conjunto_entrenamiento)

# COMMAND ----------

indexer_purpose = StringIndexer(inputCol="purpose", outputCol="purpose2").fit(conjunto_entrenamiento)
conjunto_entrenamiento = indexer_purpose.transform(conjunto_entrenamiento)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creamos el ensamblador de columnas y dejamos fuera la columna "medv" que es la que queremos  predecir

# COMMAND ----------

nombres_de_las_columnas_de_entrada = [ 'funded_amnt', 'term2', 'int_rate', 'installment', 'emp_length2', 'home_ownership2', 'annual_inc', 'verification_status2', 'purpose2', 'dti', 'delinq_2yrs' ]

# COMMAND ----------

ensamblador_de_columnas = VectorAssembler( inputCols = nombres_de_las_columnas_de_entrada, outputCol="columnas_ensambladas" )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Definimos el nombre de la columna a predecir, y el nombre de  las columnas ensambladas

# COMMAND ----------

modelo_de_regresion_lineal = (RandomForestClassifier()
  .setLabelCol("clasificacion_de_prestamo")
  .setFeaturesCol("columnas_ensambladas")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creamos un flujo de procesos

# COMMAND ----------

pasos_a_seguir = Pipeline(stages = [ ensamblador_de_columnas, modelo_de_regresion_lineal ])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Vemos los parametros con los que podemos variar este modelo

# COMMAND ----------

print( modelo_de_regresion_lineal.explainParams() )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ponemos los parametros con los que probaremos, y los valores que tomarán

# COMMAND ----------

tabla_de_parametros_a_probar = (ParamGridBuilder()
  .addGrid( modelo_de_regresion_lineal.numTrees, [ 10, 15, 20 ])
  .addGrid( modelo_de_regresion_lineal.impurity, [ 'entropy', 'gini' ])
  .build()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Definimos el evaluador que utilizaremos

# COMMAND ----------

evaluador = BinaryClassificationEvaluator(
  labelCol = "clasificacion_de_prestamo", 
  metricName="areaUnderROC"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Definimos el evaluador que utilizaremos

# COMMAND ----------

evaluacion_de_modelos_con_cross_validation = CrossValidator(
  estimator = pasos_a_seguir,             
  estimatorParamMaps = tabla_de_parametros_a_probar,   
  evaluator=evaluador,              
  numFolds = 3,                     
  seed = 12                         
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Entrenamos y obtenemos el conjunto de modelos

# COMMAND ----------

datos_de_evaluacion = evaluacion_de_modelos_con_cross_validation.fit( conjunto_entrenamiento )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Mostramos el desempeño del modelo con diferentes parametros

# COMMAND ----------

import mlflow
from mlflow import spark as mlflow_spark

# COMMAND ----------

cadena_de_resultado = ""

with mlflow.start_run():

  numero_de_paso = 0  
  mlflow.log_param( "numTrees", [ 10, 15, 20 ] )
  mlflow.log_param( "impurity", [ 'entropy', 'gini' ] )
  
  for parametros, puntuacion in zip( datos_de_evaluacion.getEstimatorParamMaps(), datos_de_evaluacion.avgMetrics ):
    
    numero_de_paso += 1
    
    parametros = "".join([parametro.name+"\t"+str(parametros[ parametro ])+"\t" for parametro in parametros ])
    puntaje = "\tPuntaje promedio: {}".format( puntuacion )
    cadena_de_resultado += parametros + "\n"
    cadena_de_resultado += puntaje + "\n"
    print( parametros )
    print( puntaje )
    
    mlflow.log_metric("areaUnderROC", puntuacion, step = numero_de_paso )
    
  with open("resultados.txt", "w") as f:
      f.write(cadena_de_resultado)
  mlflow.log_artifact("resultados.txt")
  
  mlflow_spark.log_model( datos_de_evaluacion.bestModel, "Mejor_Modelo_Random_Forest")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Obtenemos el mejor modelo de la evaluación

# COMMAND ----------

mejor_modelo = datos_de_evaluacion.bestModel

# COMMAND ----------



# COMMAND ----------


