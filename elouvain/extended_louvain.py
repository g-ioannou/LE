# from logs.logger import logger
# from elouvain.elouvain_spark import *  # pylint:disable= unused-wildcard-import

from elouvain.elouvain_spark import SparkTools
from elouvain.obj import Graph


def extended_louvain(
    first_iteration_graph: Graph,
    subgraph_depth: int,
    cosine_threshold: float,
    weight_threshold: float,
    SparkTools: SparkTools,
):
    # logger.info("------Running Louvain Extended------")
    G = first_iteration_graph

    G.nodes.cache()
    G.edges.cache()

    subgraph = G.induced_subgraph(starting_node=G.get_random_node(), depth=subgraph_depth)
    SparkTools.uncache_all()

    # sub-loop
    new_subgraph_partition = subgraph.get_partition_using_metrics(
        weight_thresh=weight_threshold, cosine_thresh=cosine_threshold
    )  # end of subloop
   

    subgraph.update_based_on_partition(new_subgraph_partition)
    # subgraph.nodes.show()
    # subgraph.edges.show()
    # Q_old = G.get_modularity()
    # Q_new = G.get_modularity(new_subgraph_partition)

    # if Q_old < Q_new:
    #     G.update_based_on_subgraph()
    # # else:
    # #     break
