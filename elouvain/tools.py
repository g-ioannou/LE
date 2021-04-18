from elouvain.obj import Graph
from elouvain.elouvain_spark import *  # pylint:disable=unused-wildcard-import

import community
import networkx as nx
import pandas



def merge_partitions(partitionA: dict, partitionB: dict) -> dict:
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


def get_random_node(Graph: Graph) -> int:
    """
    Return a random node from the Graph

    Args:
        Graph(elouvain.obj.Graph):
    Returns:
        random_node(int)
    """

    random_node = Graph.nodes.rdd.takeSample(withReplacement=False, num=1, seed=0)

    schema = StructType(
        [
            StructField("id", IntegerType(), False),
            StructField("partition", IntegerType(), False),
            StructField("vector", VectorUDT(), False),
        ]
    )

    random_node_as_df = spark.createDataFrame(random_node, schema)

    return random_node_as_df


def get_partition_using_metrics(G: nx.Graph, weight_thresh: float, cosine_thresh: float) -> dict:
    """
    Args:
        G(nx.Graph):
        weight_thresh(float):
        cosine_thresh(float):
    Returns:
        partition(dict): The best partition using modularity and thresholds.
    """

    partition = {node: node for node in G.nodes()}
    Q_old = community.modularity(partition, G)

    for node in G.nodes():
        for neighbor in list(G.neighbors(node)):
            edge_weight = G[node][neighbor]["weight"]
            cosine_sim = G[node][neighbor]["cosine_sim"]
            if edge_weight >= weight_thresh or cosine_sim >= cosine_thresh:
                partition = __move_neighbors_to_target(partition, node, neighbor)
                Q_old = community.modularity(partition, G)
            else:
                Q_new = __calculate_Q_new(G, partition, neighbor, node)
                if Q_new > Q_old:
                    partition = __move_neighbors_to_target(partition, node, neighbor)
                    Q_old = Q_new

    __add_id_as_attribute(partition, G)
    return partition


def to_communities(G: Graph, subgraph: nx.Graph, partition: dict) -> [DataFrame, DataFrame]:
    """
    Merge nodes given a partition of a subgraph's nodes. Partition keys(nodes) will be merged to the corresponding values(community).

    Args:
        G(elouvain.obj.Graph):
        subgraph(nx.Graph):

    Returns:
        new_nodes_df(pyspark.sql.DataFrame): Dataframe of nodes with their new partition.
        new_edges_df(pyspark.sql.DataFrame): Dataframe of edges connecting communities and selfloop edges.
    """
    edges_df = G.edges
    nodes_df = G.nodes

    new_nodes_df = pandas.DataFrame.from_dict(subgraph.nodes, orient="index")
    new_nodes_df = spark.createDataFrame(new_nodes_df)

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
