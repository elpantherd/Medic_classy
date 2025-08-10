import logging
import os
import sys
from datetime import datetime

def setup_logging(log_dir):
    """Sets up a centralized logger."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = f"classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file at: {log_filepath}")
