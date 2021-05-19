from elouvain.extended_louvain import extended_louvain as run_extended_louvain

# from logs.logger import logger
from elouvain.elouvain_spark import *  # pylint:disable=unused-wildcard-import
from configs.config import config

from elouvain.metrics import Metrics
from elouvain.obj import Graph


# Initializing SparkTools instance


def main():
   
    ST = SparkTools()
    
    edges_df = ST.load_csv(
        path=config.input_conf["edges"]["file_path"],
        delimiter=config.input_conf["edges"]["delimiter"],
    ).coalesce(8)

    nodes_df = ST.load_csv(
        path=config.input_conf["nodes"]["file_path"],
        delimiter=config.input_conf["nodes"]["delimiter"],
    ).coalesce(8)

    G = Graph(nodes_df, edges_df, ST)

    G.nodes = ST.reload_df(nodes_df, "nodes_df", 8).withColumn(
        "partition", F.col(config.input_conf["nodes"]["id_column_name"][0])
    )
    G.nodes = ST.calculate_vectors(G.nodes)
    ST.uncache_all()

    G.edges = ST.reload_df(edges_df, "edges_df", 8, ["src"]).withColumn("weight", F.lit(1)).cache()
    G.nodes.cache()
    G.edges = Metrics.calculate_cosine(Graph=G, features=config.input_conf["nodes"]["features"], ST=ST)

    COSINE_THRESH = config.parameters["cosine_thresh"]
    WEIGHT_THRESH = config.parameters["weight_thresh"]
    DEPTH = config.parameters["subgraph_depth"]

    run_extended_louvain(
        first_iteration_graph=G,
        subgraph_depth=DEPTH,
        cosine_threshold=COSINE_THRESH,
        weight_threshold=WEIGHT_THRESH,
        SparkTools=ST,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # logger.critical(str(e))
        print(str(e))
