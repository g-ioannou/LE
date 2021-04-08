import yaml
#from logs.logger import logger


class _Config:
    def __init__(self):
        with open(r"./configs/config.yaml") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        self.all_config = config
        self.spark_conf = self.all_config["spark"]
        self.input_conf = self.all_config["input"]
        self.parameters = self.all_config['parameters']
        # self.output_conf = self.config["output"]


#logger.info("Loading configuratiton from config.yaml")
config = _Config()