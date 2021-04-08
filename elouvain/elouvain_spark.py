from configs.config import config

# from logs.logger import logger

from pyspark.sql import SparkSession, SQLContext, DataFrame, Row, Window

from pyspark.sql.types import IntegerType, StringType, FloatType, ArrayType, StructType, StructField
from pyspark.ml.linalg import VectorUDT
import pyspark.sql.functions as F

from pyspark.ml.feature import Tokenizer, StopWordsRemover, Word2Vec

# logger.info("Initializing Spark")
spark = SparkSession.builder.getOrCreate()
config = config.spark_conf


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
        Transforms hamsterster features to vectors

        Returns:
            hamsterster_nodes(pyspark.sql.DataFrame)

        """
        hamsterster_nodes = (
            nodes_df.withColumn("Favorite_acitvity", F.regexp_replace("Favorite_activity", ",", ""))
            .select(
                F.concat_ws(" and ", nodes_df.Favorite_activity, nodes_df.Favorite_food).alias("col1"),
                "id",
                "partition",
            )
            .withColumn("col1", F.regexp_replace("col1", "Singapore,  ", ""))
        )

        tokenizer = Tokenizer(inputCol="col1", outputCol="words")
        hamsterster_nodes = tokenizer.transform(hamsterster_nodes).drop("col1")

        stopWordsRemover = StopWordsRemover(inputCol="words", outputCol="words_filtered")
        hamsterster_nodes = stopWordsRemover.transform(hamsterster_nodes).drop("words")

        word2vec = Word2Vec(vectorSize=1, minCount=1, seed=0, inputCol="words_filtered", outputCol="vector")
        model = word2vec.fit(hamsterster_nodes)
        hamsterster_nodes = model.transform(hamsterster_nodes).drop("words_filtered")

        return hamsterster_nodes

    # TODO calculate vectors for any given graph
    # def calculate_vectors(self, nodes_df: DataFrame):
    #     """
    #     Calculates feature vectors

    #     Args:
    #         nodes_df(pyspark.sql.DataFrame):
    #     """
    #     print("ok")

    @staticmethod
    def clean_hamsterster_df(nodes_df):

        cols_to_drop = [
            "Name",
            "Coloring",
            "Joined",
            "Species",
            "Gender",
            "Birthday",
            "Age",
            "Hometown",
            "Favorite_toy",
        ]

        nodes_df = nodes_df.drop(*cols_to_drop).withColumnRenamed("ID", "id")

    # TODO clean generic dataframe
    # def clean_df():