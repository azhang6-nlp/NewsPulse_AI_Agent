# File: AI_Newsletter/logger_config.py

import logging
from logging.handlers import RotatingFileHandler
import os

LOG_FILE = "ai_newsletter_runtime.log"
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
LOG_BACKUP_COUNT = 3

def setup_file_logging(log_level=logging.INFO):
    """Sets up a rotating file handler for all application logs."""
    
    # 1. Get the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Prevent duplicate handlers if called multiple times
    if any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        return

    # 2. Create and configure the file handler
    file_handler = RotatingFileHandler(
        LOG_FILE, 
        maxBytes=LOG_MAX_BYTES, 
        backupCount=LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    
    # 3. Define the log format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # 4. Add the handler
    logger.addHandler(file_handler)

    logging.info(f"--- File logging initialized: {os.path.abspath(LOG_FILE)} ---")