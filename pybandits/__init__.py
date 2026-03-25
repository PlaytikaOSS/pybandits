import logging
import sys

from loguru import logger


class InterceptHandler(logging.Handler):
    def __init__(self, level="WARNING"):
        super().__init__()
        self.setLevel(level)

    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


# Intercept numpyro logger with loguru with specific level
numpyro_handler = InterceptHandler(level="WARNING")
logging.getLogger("numpyro").handlers = [numpyro_handler]

# Configure default loguru logger (this won't affect numpyro's intercepted logs)
logger.configure(handlers=[{"sink": sys.stderr}])
