import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import warnings

from rag_project.constants import LOG_DIR


def setup_logger(name: str = "rag_project") -> logging.Logger:
    """
    Configure and return a logger instance with both file and console handlers.

    Args:
        name (str): Name of the logger. Defaults to "rag_project"

    Returns:
        logging.Logger: Configured logger instance
    """
    warnings.filterwarnings("ignore")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "rag_project.log"),
        maxBytes=10485760,  # 10MB
        backupCount=5,
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Create a default logger instance
logger = setup_logger()
