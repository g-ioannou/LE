from elouvain.extended_louvain import extended_louvain as run_extended_louvain

# from logs.logger import logger
from elouvain.elouvain_spark import *
from configs.config import config

from elouvain.metrics import Metrics
from elouvain.obj import Graph


def main():

    edges_df = SparkTools.load_csv(
        path=config.input_conf["edges"]["file_path"],
        delimiter=config.input_conf["edges"]["delimiter"],
    ).withColumn("weight", F.lit(1))

    nodes_df = SparkTools.load_csv(
        path=config.input_conf["nodes"]["file_path"],
        delimiter=config.input_conf["nodes"]["delimiter"],
    )

    G = Graph(nodes_df, edges_df)

    G.nodes = SparkTools.calculate_hamsterster_vectors(G.nodes)

    G.edges = Metrics.calculate_cosine(G)

    COSINE_THRESH = config.parameters["cosine_thresh"]
    WEIGHT_THRESH = config.parameters["weight_thresh"]
    DEPTH = config.parameters["subgraph_depth"]

    run_extended_louvain(
        first_iteration_graph=G,
        subgraph_depth=DEPTH,
        cosine_threshold=COSINE_THRESH,
        weight_threshold=WEIGHT_THRESH,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # logger.critical(str(e))
        print(str(e))