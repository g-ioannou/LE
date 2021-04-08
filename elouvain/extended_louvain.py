# from logs.logger import logger
from elouvain.elouvain_spark import *  # pylint:disable= unused-wildcard-import

from elouvain.tools import *  # pylint:disable= unused-wildcard-import
from elouvain.obj import Graph


def extended_louvain(
    first_iteration_graph: Graph,
    subgraph_depth: int,
    cosine_threshold: float,
    weight_threshold: float,
):
    # logger.info("------Running Louvain Extended------")
    G = first_iteration_graph

    # while True:
    subgraph_nodes, subgraph_edges = G.induced_subgraph(starting_node=get_random_node(G), depth=subgraph_depth)

    subgraph = Graph(subgraph_nodes, subgraph_edges).to_nx()

    subgraph_partition = get_partition_using_metrics(
        G=subgraph,
        weight_thresh=weight_threshold,
        cosine_thresh=cosine_threshold,
    )

    print(subgraph_partition)