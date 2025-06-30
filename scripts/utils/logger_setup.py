import os
import logging
from datetime import datetime


def setup_logging(log_dir: str = "wiener_analysis_logs") -> logging.Logger:
    """
    Set up logging for the Wiener index analysis.
    Args:
        log_dir: Directory where log files will be stored. Defaults to "wiener_analysis_logs".
    Returns:
        A configured logger instance.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"wiener_analysis_{timestamp}.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger
