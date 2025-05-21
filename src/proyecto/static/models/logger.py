import logging

class LoggerConfig:
    @staticmethod
    def configure_logging():
        logger = logging.getLogger('DataCollector')
        handler = logging.FileHandler('src/proyecto/static/models/collector.log', encoding="utf-8")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

LoggerConfig.configure_logging()
