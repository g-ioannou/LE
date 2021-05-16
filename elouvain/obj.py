from elouvain.elouvain_spark import *  # pylint:disable=unused-wildcard-import
import community
import networkx as nx
import json
import numpy
import community
import networkx as nx
import pandas


class Graph:
    """
    Graph used in the Extended Louvain' s Algorithm.
    """

    def __init__(self, nodes_df: DataFrame, edges_df: DataFrame, ST: SparkTools):
        self.nodes = nodes_df
        self.edges = edges_df
        self.ST = ST

    def update_based_on_partition(self, new_partition):
        """
        Updates self.nodes.partition based on new_partition.values().

        Updates self.edges based on the partition in the following way:
            -Make self-loops for nodes in the same partition and delete the old edges connecting them (merge nodes -> hypernode).
            -Updates old edges connecting nodes of different partitions to new edges connecting nodes of different partitions to hypernodes.

        """
        partition_zipped = list(zip(new_partition.keys(), new_partition.values()))

        new_partition_df = (
            self.ST.spark.createDataFrame(partition_zipped)
            .withColumnRenamed("_1", "id_temp")
            .withColumnRenamed("_2", "new_partition")
        )

        def vec2array(vector):
            vector = Vectors.dense(vector)
            array = list([float(x) for x in vector])
            return array

        vec2array_udf = F.udf(vec2array, ArrayType(FloatType()))

        # update partition column based on new_partition dict
        new_nodes = (
            self.nodes.join(new_partition_df, on=self.nodes.id == new_partition_df.id_temp)
            .drop("id_temp", "partition")
            .withColumnRenamed("new_partition", "partition")
            .withColumn("vector", vec2array_udf("vector"))
        )

        # find median for vectors of nodes in the same partition
        vectors_in_partitions = new_nodes.groupBy("partition").pivot("id").sum("vector")
        vectors_in_partitions.show()

        vectors_in_partitions = vectors_in_partitions.withColumn(
            "vec_comb", F.sort_array(F.array([x for x in vectors_in_partitions.columns if x != "partition"]))
        ).withColumn("vec_comb_cleaned", F.expr("filter(vec_comb,x->x is not null)"))

        udf_median = F.udf(f=lambda c: numpy.median(numpy.array(c)).item(), returnType=FloatType())
        vectors_in_partitions = (
            (
                vectors_in_partitions.select(
                    [c for c in vectors_in_partitions.columns if c in {"partition", "vec_comb_cleaned"}]
                )
                .withColumnRenamed("vec_comb_cleaned", "vec_comb")
                .withColumn("vector_temp", udf_median("vec_comb"))
            )
            .drop("vec_comb")
            .withColumnRenamed("partition", "partition_temp")
        )

        # update edges based on the merges given by the partition dict
        new_edges = (
            self.edges.join(new_nodes, on=self.edges.src == new_nodes.id)
            .drop("id", "vector")
            .withColumnRenamed("partition", "src_partition")
            .join(new_nodes, on=self.edges.dst == new_nodes.id)
            .drop("id", "vector")
            .withColumnRenamed("partition", "dst_partition")
            .groupBy("src_partition", "dst_partition")
            .sum()
            .drop("sum(src)", "sum(dst)", "sum(src_partition)", "sum(dst_partition)")
            .withColumnRenamed("sum(weight)", "weight")
            .withColumnRenamed("src_partition", "src")
            .withColumnRenamed("dst_partition", "dst")
            .withColumn("src_dst", F.sort_array(F.array("src", "dst")))
            .withColumn("weight", F.sum("weight").over(Window.partitionBy("src_dst")))
            .dropDuplicates(["src_dst"])
            .drop("src_dst")
        )

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
            partition = self.merge_partitions(self.get_current_partition(), partition)

        modularity = community.modularity(partition, self.to_nx())
        return modularity

    def to_nx(self) -> nx.Graph:
        """

        Args:

        Returns:
            G(networkx.Graph): NetworkX graph from nodes and edges dataframes
        """
        node_attr = self.nodes.toPandas().set_index("id").to_dict("index")
        G = nx.from_pandas_edgelist(
            self.edges.toPandas(), source="src", target="dst", edge_attr=["weight", "cosine_sim"]
        )
        nx.set_node_attributes(G, node_attr)
        return G

    def induced_subgraph(
        self,
        starting_node: DataFrame,
        depth: int,
    ):
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
        subgraph_edges = self.ST.spark.sql(
            "SELECT * \
            FROM edges_df \
            WHERE \
                edges_df.src IN(SELECT * FROM subgraph_nodes) \
            AND edges_df.dst IN (SELECT * FROM subgraph_nodes)"
        )

        self.nodes.createOrReplaceTempView("nodes_df")
        subgraph_nodes = self.ST.spark.sql("SELECT * FROM nodes_df WHERE nodes_df.id IN (SELECT * FROM subgraph_nodes)")

        return Graph(subgraph_nodes, subgraph_edges, self.ST)

    # def update_based_on_subgraph(self, subgraph:Graph) -> Graph:
    #     # ...

    def merge_partitions(self, partitionA: dict, partitionB: dict) -> dict:
        """
        Returns a partition after updating partitionA based on partitionB.

        Args:
            partitionA(dict):
        Returns:
        """

        new_partition = partitionA.copy()
        for node in partitionB:
            new_partition[node] = partitionB[node]

        return new_partition

    def get_partition_using_metrics(self, weight_thresh: float, cosine_thresh: float) -> dict:
        """
        Args:
            G(nx.Graph):
            weight_thresh(float):
            cosine_thresh(float):
        Returns:
            partition(dict): The best partition using modularity and thresholds.
        """
        G = self.to_nx()
        partition = {node: node for node in G.nodes()}
        Q_old = community.modularity(partition, G)

        for node in G.nodes():
            for neighbor in list(G.neighbors(node)):
                edge_weight = G[node][neighbor]["weight"]
                cosine_sim = G[node][neighbor]["cosine_sim"]
                if edge_weight >= weight_thresh or cosine_sim >= cosine_thresh:
                    partition = Graph.__move_neighbors_to_target(partition, node, neighbor)
                    Q_old = community.modularity(partition, G)
                else:
                    Q_new = Graph.__calculate_Q_new(G, partition, neighbor, node)
                    if Q_new > Q_old:
                        partition = Graph.__move_neighbors_to_target(partition, node, neighbor)
                        Q_old = Q_new

        Graph.__add_id_as_attribute(partition, G)
        return partition

    def to_communities(self, subgraph: nx.Graph, partition: dict) -> List[DataFrame]:
        """
        Merge nodes given a partition of a subgraph's nodes. Partition keys(nodes) will be merged to the corresponding values(community).

        Args:
            G(elouvain.obj.Graph):
            subgraph(nx.Graph):

        Returns:
            new_nodes_df(pyspark.sql.DataFrame): Dataframe of nodes with their new partition.
            new_edges_df(pyspark.sql.DataFrame): Dataframe of edges connecting communities and selfloop edges.
        """
        edges_df = self.edges
        nodes_df = self.nodes

        new_nodes_df = pandas.DataFrame.from_dict(subgraph.nodes, orient="index")
        new_nodes_df = self.ST.spark.createDataFrame(new_nodes_df)

        new_edges_df = (
            edges_df.join(nodes_df, on=edges_df.src == nodes_df.id)
            .drop("id", "cosine_sim", "vector")
            .withColumnRenamed("partition", "src_partition")
            .join(nodes_df, on=edges_df.dst == nodes_df.id)
            .drop("id", "cosine_sim", "vector")
            .withColumnRenamed("partition", "dst_partition")
            .groupBy("src_partition", "dst_partition")
            .sum()
            .drop("sum(src)", "sum(dst)", "sum(src_partition)", "sum(dst_partition)")
            .withColumnRenamed("sum(weight)", "weight")
            .withColumnRenamed("src_partition", "src")
            .withColumnRenamed("dst_partition", "dst")
            .withColumn("src_dst", F.sort_array(F.array("src", "dst")))
            .withColumn("weight", F.sum("weight").over(Window.partitionBy("src_dst")))  # pylint:disable=no-member
            .dropDuplicates(["src_dst"])
            .drop("src_dst")
        )

        print(new_edges_df.count())

        return new_edges_df, new_nodes_df

    @staticmethod
    def __add_id_as_attribute(partition: dict, G: nx.Graph):
        """
        Simply adds the node ID as a node attribute.

        Args:
            partition(dict):
            G(nx.Graph):
        """
        for node in partition:
            attrs = {}
            attrs[node] = {"id": node, "partition": partition[node]}
            nx.set_node_attributes(G, attrs)

    @staticmethod
    def __calculate_Q_new(G: nx.Graph, partition: dict, node: int, target: int) -> float:
        """
        Returns the new modularity after moving 'node' to the partition of 'target'

        Args:
            G(nx.Graph):
            partition(dict):
            node(int): Node to be moved.
            target(int):
        """
        temp_partition = partition.copy()
        temp_partition[node] = temp_partition[target]
        return community.modularity(temp_partition, G)

    @staticmethod
    def __move_neighbors_to_target(partition: dict, target: int, neighbor: int) -> dict:
        """
        Moves all nodes that belong in the same partition as 'neighbor' to the partition of 'target'

        Args:
            G(nx.Graph):
            partition(dict): Current iteration's partition.
            target(int): The node's partition all 'neighbor' nodes will be moved to.
            neighbor(int):

        """
        nodes_in_neighbors_community = [
            node_key for node_key, node_partition in partition.items() if node_partition == neighbor
        ]
        for contained_node in nodes_in_neighbors_community:
            partition[contained_node] = partition[target]

        return partition

    def get_random_node(self) -> int:
        """
        Return a random node from the Graph

        Args:
            Graph(elouvain.obj.Graph):
        Returns:
            random_node(int)
        """
        Graph.nodes = Graph.nodes.select("id", "partition", "vector")
        random_node = Graph.nodes.rdd.takeSample(withReplacement=False, num=1, seed=0)

        schema = StructType(
            [
                StructField("id", IntegerType(), False),
                StructField("partition", IntegerType(), False),
                StructField("vector", VectorUDT(), False),
            ]
        )

        random_node_as_df = self.ST.spark.createDataFrame(random_node, schema)
        return random_node_as_df
