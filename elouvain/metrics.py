from os import terminal_size
from pyspark.sql.functions import lit, sqrt
from pyspark.mllib.linalg import DenseVector
from elouvain.elouvain_spark import *  # pylint:disable=unused-wildcard-import
from elouvain.obj import Graph
import numpy

from configs.config import config


class Metrics:
    @staticmethod
    def calculate_cosine(Graph: Graph, features: list, ST: SparkTools) -> DataFrame:
        """
        Calculates cosine similarities of neighbor nodes. Assigns the similarity value to the corresponding edge.

        Args:
            Graph(elouvain.obj.Graph): Graph object. Graph.nodes must contain a vector column corresponding to each node's features vector.
        Returns:
            edges_with_cosine(pyspark.sql.DataFrame): edges dataframe with 'cosine_sim' column

        """
        number_of_features = len(features)

        edges_vectors_comb = (
            Graph.edges.sortWithinPartitions("src")
            .alias("edges")
            .join(Graph.nodes.alias("nodes"), on=F.col("edges.src") == F.col("nodes.id"))
            .drop("id")
            .withColumnRenamed("vector", "vector_src")
            .sortWithinPartitions("dst")
            .cache()
            .join(Graph.nodes.alias("nodes"), on=F.col("edges.dst") == F.col("nodes.id"))
            .drop("id")
            .withColumnRenamed("vector", "vector_dst")
            .select(["src", "dst", "weight", "vector_src", "vector_dst"])
            .cache()
        )

        edges_vectors_comb = (
            edges_vectors_comb.withColumn("vector_src_array_tmp", MLF.vector_to_array("vector_src"))
            .select(
                ["src", "dst", "weight", "vector_dst", "vector_src"]
                + [F.col("vector_src_array_tmp")[i] for i in range(number_of_features)]
            )
            .cache()
        )

        edges_vectors_comb = (
            edges_vectors_comb.withColumn("vector_dst_array_tmp", MLF.vector_to_array("vector_dst"))
            .select(
                ["src", "dst", "weight", "vector_src", "vector_dst"]
                + [col for col in edges_vectors_comb.columns if "array" in col]
                + [F.col("vector_dst_array_tmp")[i] for i in range(number_of_features)]
            )
            .withColumn("dot", lit(0))
            .withColumn("norm_src", lit(0))
            .withColumn("norm_dst", lit(0))
            .cache()
        )

        # calculate dot product
        for feature_index in range(0, number_of_features):
            src_features_col_name = "vector_src_array_tmp[" + str(feature_index) + "]"
            dst_features_col_name = "vector_dst_array_tmp[" + str(feature_index) + "]"
            edges_vectors_comb = edges_vectors_comb.withColumn(
                "dot", F.col("dot") + F.col(src_features_col_name) * F.col(dst_features_col_name)
            ).cache()

        # calculate norm of vectors
        for feature_index in range(0, number_of_features):
            src_features_col_name = "vector_src_array_tmp[" + str(feature_index) + "]"
            dst_features_col_name = "vector_dst_array_tmp[" + str(feature_index) + "]"
            edges_vectors_comb = (
                edges_vectors_comb.withColumn(
                    "norm_src", F.col("norm_src") + F.col(src_features_col_name) * F.col(src_features_col_name)
                )
                .withColumn("norm_dst", F.col("norm_dst") + F.col(dst_features_col_name) * F.col(dst_features_col_name))
                .cache()
            )

        edges_vectors_comb = (
            edges_vectors_comb.withColumn("norm_src", sqrt("norm_src"))
            .withColumn("norm_dst", sqrt("norm_dst"))
            .withColumn("cosine_sim", F.col("dot") / (F.col("norm_src") * F.col("norm_dst")))
            .select(["src", "dst", "weight", "cosine_sim"])
            .cache()
        )

        # edges with cosine
        Graph.edges = edges_vectors_comb.drop("vector_src").drop("vector_dst").drop("partition")

        return Graph.edges
