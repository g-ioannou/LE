from networkx.classes.function import nodes
from configs.config import config

# from logs.logger import logger

from pyspark.sql import SparkSession, SQLContext, DataFrame, Row, Window

from pyspark.sql.types import IntegerType, StringType, FloatType, ArrayType, StructType, StructField
from pyspark.ml.linalg import VectorUDT, Vectors
import pyspark.sql.functions as F
import pyspark.ml.functions as MLF

from pyspark.ml.feature import Tokenizer, StopWordsRemover, Word2Vec, VectorAssembler

# logger.info("Initializing Spark")
spark = SparkSession.builder.getOrCreate()


class SparkTools:
    """
    Extended Louvain spark tools.
    """

    @staticmethod
    def load_csv(delimiter, path: str, has_header: bool = True, infer_schema: bool = True) -> DataFrame:
        """
        Loads a .csv to a dataframe.

        Args:
            path (str): Path to .csv

        Returns:
            DataFrame
        """

        return spark.read.option("delimiter", delimiter).csv(path, header=has_header, inferSchema=infer_schema)

    @staticmethod
    def calculate_hamsterster_vectors(nodes_df: DataFrame) -> DataFrame:
        """
        Transforms given features to vectors

        Returns:
            hamsterster_nodes(pyspark.sql.DataFrame)

        """
        id_col = config.input_conf["nodes"]["id_column_name"]
        features = config.input_conf["nodes"]["features"]
        cols = id_col + features

        nodes_df = nodes_df.select([col for col in cols]).withColumnRenamed(id_col[0], "id")

        for feature in features:

            spec = Window.partitionBy().orderBy(feature)
            feature_values = (
                nodes_df.select(feature)
                .distinct()
                .withColumn(str(feature + "_value"), F.row_number().over(spec))
                .withColumnRenamed(feature, str(feature + "_temp"))
            )
            
            nodes_df = (
                nodes_df.join(feature_values, on=nodes_df[feature] == feature_values[str(feature + "_temp")])
                .drop(feature)
                .drop(str(feature + "_temp"))
            )

        
        assembler = VectorAssembler(
            inputCols=[col for col in nodes_df.columns if col != 'id'],
            outputCol="vector",
        )

        nodes_df = assembler.transform(nodes_df)
        nodes_df = nodes_df.select([col for col in nodes_df.columns if "_value" not in col])
        
        return nodes_df

