import logging
import sys
from pathlib import Path

def setup_logging():
    # Create folder to save images + metadata + logs (if not already present)
    Path("insect-detect/data").mkdir(parents=True, exist_ok=True)

    # Create logger and write info + error messages to log file
    logging.basicConfig(filename="insect-detect/data/script_log.log", encoding="utf-8",
                        format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO)
    logger = logging.getLogger()
    sys.stderr.write = logger.error
    # Inform that logging has been configured
    logging.info("Logging is configured.")
    
    return logger
