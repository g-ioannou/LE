from elouvain.elouvain_spark import *  # pylint:disable=unused-wildcard-import
from elouvain.obj import Graph

import numpy


class Metrics:

    # TODO This should work for config.nodes.input_columns
    @staticmethod
    def calculate_cosine(Graph: Graph) -> DataFrame:
        """
        Calculates cosine similarities of neighbor nodes. Assigns the similarity value to the corresponding edge.

        Args:
            Graph(elouvain.obj.Graph): Graph object.
        Returns:
            edges_with_cosine(pyspark.sql.DataFrame): edges dataframe with 'cosine_sim' column

        """

        udf_cosine_sim = F.udf(
            f=lambda a, b: float(numpy.dot(a, b) / numpy.linalg.norm(a) * numpy.linalg.norm(b)), returnType=FloatType()
        )

        edges_vectors_comb = (
            Graph.edges.join(Graph.nodes, on=Graph.edges.src == Graph.nodes.id)
            .drop("id")
            .withColumnRenamed("vector", "vector_src")
            .join(Graph.nodes, on=Graph.edges.dst == Graph.nodes.id)
            .drop("id")
            .withColumnRenamed("vector", "vector_dst")
        )

        # edges with cosine
        Graph.edges = (
            edges_vectors_comb.withColumn(
                "cosine_sim",
                udf_cosine_sim(  # pylint: disable=redundant-keyword-arg
                    edges_vectors_comb.vector_src, edges_vectors_comb.vector_dst
                ),
            )
            .drop("vector_src")
            .drop("vector_dst")
            .drop("partition")
        )

        return Graph.edges
