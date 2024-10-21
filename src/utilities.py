import logging
import logging.config 
import yaml 
from datetime import datetime

def get_config():
    with open("src/params.yaml", "r") as config: 
        config = yaml.safe_load(config)

    return config

def create_logger():
    log_filename = f"logs/log_{datetime.now().strftime('%Y-%m-%d__%H-%M-%S')}.log"
    logger = logging.getLogger("ML_log")
    config = get_config()
    config['logger']['handlers']['file']['filename'] = log_filename
    logging.config.dictConfig(config=config['logger'])
    
    return logger


if __name__ == '__main__':

    logger = create_logger()
    config = get_config()
    logger.debug("debug_message")
    logger.info("info_message")
