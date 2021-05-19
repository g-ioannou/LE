from __future__ import annotations

from networkx.classes.function import nodes
from pyspark.context import SparkContext
from configs.config import config

# from logs.logger import logger

from pyspark.sql import SparkSession, SQLContext, DataFrame, Row, Window
from pyspark import SparkConf
from pyspark.sql.types import IntegerType, StringType, FloatType, ArrayType, StructType, StructField
from pyspark.ml.linalg import VectorUDT, Vectors
import pyspark.sql.functions as F
import pyspark.ml.functions as MLF
from pyspark.ml.feature import Tokenizer, StopWordsRemover, Word2Vec, VectorAssembler
from typing import List
import shutil
import os
from pathlib import Path


class SparkTools:
    """
    Extended Louvain spark tools.
    """

    def __init__(self) -> None:
        self.conf = SparkConf()
        [self.conf.set(str(key), str(value)) for key, value in config.spark_conf["config"].items()]
        self.id_col = config.input_conf["nodes"]["id_column_name"]
        self.features_list = config.input_conf["nodes"]["features"]
        self.temp_df_folder = config.spark_conf["dirs"]["temp_folder"]
        self.project_path = str(Path(__file__).parent.parent) + "/"

        self.__clear_tmp_folder()

        self.spark = SparkSession.builder.config(conf=self.conf).getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")

    def calculate_vectors(self, nodes_df: DataFrame) -> DataFrame:
        """
        Transforms given features to vectors

        Returns:
            hamsterster_nodes(pyspark.sql.DataFrame)

        """

        cols = self.id_col + ["partition"] + self.features_list

        nodes_df = nodes_df.select([col for col in cols]).withColumnRenamed(self.id_col[0], "id").cache()

        for feature in self.features_list:

            spec = Window.partitionBy().orderBy(feature)
            feature_values = (
                nodes_df.select(feature)
                .distinct()
                .withColumn(str(feature + "_value"), F.row_number().over(spec))
                .withColumnRenamed(feature, str(feature + "_temp"))
                .coalesce(8)
                .cache()
            )
            # nodes_df = self.reload_df(nodes_df, "nodes_df", 4, [feature]).sortWithinPartitions(feature)
            nodes_df = (
                nodes_df.join(feature_values, on=nodes_df[feature] == feature_values[str(feature + "_temp")])
                .drop(feature)
                .drop(str(feature + "_temp"))
                .coalesce(8)
                .cache()
            )

            
        
        # assembler = VectorAssembler(
        #     inputCols=[col for col in nodes_df.columns if col != "id"],
        #     outputCol="vector",
        # )
        # nodes_df = assembler.transform(nodes_df).cache()

        # nodes_df = nodes_df.select([col for col in nodes_df.columns if "_value" not in col])
        

        return nodes_df

    def reload_df(
        self,
        df: DataFrame,
        name: str,
        num_partitions: int = None,
        partition_cols: List[str] = None,
    ) -> DataFrame:
        """
        Saves and reloads a dataframe.
        """
        parquet_path = self.temp_df_folder + name + ".parquet"

        if os.path.exists(self.project_path + parquet_path):
            parquet_path_tmp = parquet_path + ".tmp"
            self.save_parquet(df, parquet_path_tmp, num_partitions, partition_cols, True)

        else:
            self.save_parquet(df, parquet_path, num_partitions, partition_cols)

        df = self.load_parquet(parquet_path).cache()
        return df

    def save_parquet(
        self,
        df: DataFrame,
        parquet_path: str,
        num_partitions: int = None,
        partition_cols: List[str] = None,
        is_temp: bool = False,
        mode: str = "overwrite",
    ):
        """
        Saves a dataframe to parquet file.

        Args:
            df (pyspark.sql.DataFrame):
            path(str): Path to parquet
            num_partitions(int):
            partition_cols(List[str])
        """

        if num_partitions and not partition_cols:
            df.coalesce(num_partitions).write.parquet(parquet_path)
        elif num_partitions and partition_cols:
            df.repartition(num_partitions, *partition_cols).write.parquet(parquet_path)
        elif partition_cols and not num_partitions:
            df.repartition(*partition_cols).write.parquet(parquet_path)
        else:
            df.coalesce(1).write.parquet(parquet_path)

        if ".tmp" in parquet_path:
            parquet_path = parquet_path[:-4]
            shutil.rmtree(self.project_path + parquet_path)
            os.rename(self.project_path + parquet_path + ".tmp", self.project_path + parquet_path)

    def load_parquet(self, path: str) -> DataFrame:
        """
        Loads a parquet file.

        Args:
            path (str): Path to .parquet

        Returns:
            pyspark.SQL.DataFrame
        """
        df = self.spark.read.format("parquet").load(path)

        return df

    def load_csv(self, delimiter, path: str, has_header: bool = True, infer_schema: bool = True) -> DataFrame:
        """
        Loads a .csv to a dataframe.

        Args:
            path (str): Path to .csv

        Returns:
            DataFrame
        """
        return self.spark.read.option("delimiter", delimiter).csv(path, header=has_header, inferSchema=infer_schema)

    def uncache_all(self):
        [rdd.unpersist() for rdd in list(self.spark.sparkContext._jsc.getPersistentRDDs().values())]
        self.spark.catalog.clearCache()

    def __clear_tmp_folder(self):
        shutil.rmtree(self.project_path + self.temp_df_folder)

    def get_cached_data(self):
        [
            print(
                {
                    "name": s.name(),
                    "numPartitions": s.numPartitions(),
                    "numCachedPartitions": s.numCachedPartitions(),
                    "id": s.id(),
                    "isCached": s.isCached(),
                }
            )
            for s in self.spark.sparkContext._jsc.sc().getRDDStorageInfo()
        ]
        for s in self.spark.sparkContext._jsc.sc().getRDDStorageInfo():
            print(s.id)
