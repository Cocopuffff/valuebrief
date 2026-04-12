import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging(log_level=logging.INFO, log_file="logs/valuebrief.log"):
    """Sets up logging for the entire project."""
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers if any (to avoid duplicate logs)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - [%(name)s] [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File Handler (Rotating: 5MB per file, max 5 files)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5*1024*1024, backupCount=10
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.info(f"Logging initialized. Level: {logging.getLevelName(log_level)}, File: {log_file}")

def get_logger(name):
    """Returns a logger with the specified name."""
    return logging.getLogger(name)
