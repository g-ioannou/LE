# from logs.logger import logger
from elouvain.elouvain_spark import *  # pylint:disable= unused-wildcard-import

import elouvain.tools as tools
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

    subgraph = G.induced_subgraph(starting_node=tools.get_random_node(G),
                                  depth=subgraph_depth)

    # sub-loop
    new_subgraph_partition = tools.get_partition_using_metrics(
        G=subgraph.to_nx(),
        weight_thresh=weight_threshold,
        cosine_thresh=cosine_threshold,
    )  # end of subloop

    subgraph.update_based_on_partition(new_subgraph_partition)

    Q_old = G.get_modularity()
    Q_new = G.get_modularity(new_subgraph_partition)

    hello = x

    # if Q_old < Q_new:
    #     G.update_based_on_subgraph()
    # # else:
    # #     break
