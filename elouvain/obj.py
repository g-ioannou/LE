from elouvain.elouvain_spark import *  # pylint:disable=unused-wildcard-import

import networkx as nx


class Graph:
    """
    Graph used in the Extended Louvain's Algorithm.
    """

    def __init__(self, nodes_df: DataFrame, edges_df: DataFrame):
        self.nodes = nodes_df.withColumn("partition", F.col("ID"))  # pylint:disable=no-member
        self.edges = edges_df
        self.number_of_nodes = nodes_df.count()
        self.number_of_edges = edges_df.count()
        self.pandas_edges = self.edges.toPandas()
        self.pandas_nodes = self.nodes.toPandas()


    def to_nx(self) -> nx.Graph:
        """

        Args:

        Returns:
            G(networkx.Graph): NetworkX graph from nodes and edges dataframes
        """
        node_attr = self.pandas_nodes.set_index("id").to_dict("index")
        G = nx.from_pandas_edgelist(self.pandas_edges, source="src", target="dst", edge_attr=["weight", "cosine_sim"])
        nx.set_node_attributes(G, node_attr)
        return G

    def induced_subgraph(
        self,
        starting_node: DataFrame,
        depth: int,
    ) -> [DataFrame, DataFrame]:
        """
        Calculates a subgraph from the starting node up to k-depth neighbors.

        Args:
            starting_node(pyspark.sql.Dataframe): A random node in the graph

        Returns: An induced subgraph from the given graph.
            subgraph_nodes(pyspark.sql.DataFrame): Subgraph nodes.(no data)
            subgraph_edges(pyspark.sql.DataFrame): Subgraph edges.(no data)
        """

        subgraph_nodes = starting_node.select("id")

        for depth in range(1, depth):  # pylint: disable=unused-variable
            subgraph_edges = (
                subgraph_nodes.join(self.edges, on=subgraph_nodes["id"] == self.edges["dst"], how="inner")
                .drop("dst")
                .withColumnRenamed("src", "dst")
                .union(
                    subgraph_nodes.join(self.edges, on=subgraph_nodes["id"] == self.edges["src"], how="inner").drop(
                        "src"
                    )
                )
                .distinct()
            )

            subgraph_nodes = subgraph_nodes.union(
                subgraph_edges.select("dst").withColumnRenamed("dst", "id")
            ).distinct()

        subgraph_nodes.createOrReplaceTempView("subgraph_nodes")

        self.edges.createOrReplaceTempView("edges_df")
        subgraph_edges = spark.sql(
            "SELECT * \
            FROM edges_df \
            WHERE \
                edges_df.src IN(SELECT * FROM subgraph_nodes) \
            AND edges_df.dst IN (SELECT * FROM subgraph_nodes)"
        )

        self.nodes.createOrReplaceTempView("nodes_df")
        subgraph_nodes = spark.sql("SELECT * FROM nodes_df WHERE nodes_df.id IN (SELECT * FROM subgraph_nodes)")

        return subgraph_nodes, subgraph_edges
