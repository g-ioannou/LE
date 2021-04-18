from elouvain.elouvain_spark import *  # pylint:disable=unused-wildcard-import
import elouvain.tools as tools

import community
import networkx as nx
import json
import numpy


class Graph:
    """
    Graph used in the Extended Louvain's Algorithm.
    """
    def __init__(self, nodes_df: DataFrame, edges_df: DataFrame):
        self.nodes = nodes_df.withColumn("partition", F.col("ID"))  # pylint:disable=no-member
        self.edges = edges_df
        self.number_of_nodes = nodes_df.count()
        self.number_of_edges = edges_df.count()

    def update_based_on_partition(self, new_partition):
        """
        Updates self.nodes.partition based on new_partition.values().

        Updates self.edges based on the partition in the following way:
            -Make self-loops for nodes in the same partition and delete the old edges connecting them (merge nodes -> hypernode).
            -Updates old edges connecting nodes of different partitions to new edges connecting nodes of different partitions to hypernodes.

        """
        partition_zipped = list(
            zip(new_partition.keys(), new_partition.values()))

        new_partition_df = (
            spark.createDataFrame(partition_zipped).withColumnRenamed(
                "_1", "id_temp").withColumnRenamed("_2", "new_partition"))

        # update partition column based on new_partition dict
        new_nodes = (self.nodes.join(
            new_partition_df,
            on=self.nodes.id == new_partition_df.id_temp).drop([
                "id_temp,partition"
            ]).withColumnRenamed("new_partition", "partition"))

        # find median for vectors of nodes in the same partition
        vectors_in_partitions = new_nodes.groupBy("partition").pivot("id").sum(
            "vector")

        vectors_in_partitions = vectors_in_partitions.withColumn(
            "vec_comb",
            F.sort_array(
                F.array([
                    x for x in vectors_in_partitions.columns
                    if x != "partition"
                ]))).withColumn("vec_comb_cleaned",
                                F.expr("filter(vec_comb,x->x is not null)"))

        udf_median = F.udf(f=lambda c: numpy.median(numpy.array(c)).item(),
                           returnType=FloatType())
        vectors_in_partitions = ((vectors_in_partitions.select([
            c for c in vectors_in_partitions.columns
            if c in {"partition", "vec_comb_cleaned"}
        ]).withColumnRenamed("vec_comb_cleaned", "vec_comb").withColumn(
            "vector_temp",
            udf_median("vec_comb"))).drop("vec_comb").withColumnRenamed(
                "partition", "partition_temp"))

        # update edges based on the merges given by the partition dict
        new_edges = (self.edges.join(
            new_nodes, on=self.edges.src == new_nodes.id).drop(
                "id",
                "vector").withColumnRenamed("partition", "src_partition").join(
                    new_nodes, on=self.edges.dst == new_nodes.id).drop(
                        "id", "vector").withColumnRenamed(
                            "partition", "dst_partition").groupBy(
                                "src_partition", "dst_partition").sum().drop(
                                    "sum(src)", "sum(dst)",
                                    "sum(src_partition)",
                                    "sum(dst_partition)").withColumnRenamed(
                                        "sum(weight)",
                                        "weight").withColumnRenamed(
                                            "src_partition",
                                            "src").withColumnRenamed(
                                                "dst_partition",
                                                "dst").withColumn(
                                                    "src_dst",
                                                    F.sort_array(
                                                        F.array("src",
                                                                "dst"))).
                     withColumn(
                         "weight",
                         F.sum("weight").over(
                             Window.partitionBy("src_dst"))).dropDuplicates(
                                 ["src_dst"]).drop("src_dst"))

    def get_current_partition(self) -> dict:
        """
        Returns the graph's current partition.

        Args:

        Returns:
            partition(dict):
        """
        partition = self.nodes.select("id", "partition").rdd.collectAsMap()
        return partition

    def get_modularity(self, partition=None):
        """
        Returns the modularity of a graph.

        Args:
            partition(dict): A dict containing a new partition or a subset of repartitioned nodes.

        Returns:
            modularity(float):
        """
        if partition == None:
            partition = self.get_current_partition()
        else:
            partition = tools.merge_partitions(self.get_current_partition(),
                                               partition)

        modularity = community.modularity(partition, self.to_nx())
        return modularity

    def to_nx(self) -> nx.Graph:
        """

        Args:

        Returns:
            G(networkx.Graph): NetworkX graph from nodes and edges dataframes
        """
        node_attr = self.nodes.toPandas().set_index("id").to_dict("index")
        G = nx.from_pandas_edgelist(self.edges.toPandas(),
                                    source="src",
                                    target="dst",
                                    edge_attr=["weight", "cosine_sim"])
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
            subgraph_edges = (subgraph_nodes.join(
                self.edges,
                on=subgraph_nodes["id"] == self.edges["dst"],
                how="inner").drop("dst").withColumnRenamed("src", "dst").union(
                    subgraph_nodes.join(
                        self.edges,
                        on=subgraph_nodes["id"] == self.edges["src"],
                        how="inner").drop("src")).distinct())

            subgraph_nodes = subgraph_nodes.union(
                subgraph_edges.select("dst").withColumnRenamed(
                    "dst", "id")).distinct()

        subgraph_nodes.createOrReplaceTempView("subgraph_nodes")

        self.edges.createOrReplaceTempView("edges_df")
        subgraph_edges = spark.sql("SELECT * \
            FROM edges_df \
            WHERE \
                edges_df.src IN(SELECT * FROM subgraph_nodes) \
            AND edges_df.dst IN (SELECT * FROM subgraph_nodes)")

        self.nodes.createOrReplaceTempView("nodes_df")
        subgraph_nodes = spark.sql(
            "SELECT * FROM nodes_df WHERE nodes_df.id IN (SELECT * FROM subgraph_nodes)"
        )

        return Graph(subgraph_nodes, subgraph_edges)

    # def update_based_on_subgraph(self, subgraph:Graph) -> Graph:
    #     # ...
