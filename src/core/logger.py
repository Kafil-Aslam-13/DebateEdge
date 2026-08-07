# here i will do centralized logging
"""Centralized logger for DebateEdge
every module calls get_logger(__name__) - never configures logging
itself. One Format,One Place."""

import logging , os , sys
from datetime import datetime

_LOG_DIR="logs"
os.makedirs(_LOG_DIR,exist_ok=True)

_LOG_FILE = os.path.join(_LOG_DIR,f"debateedge_{datetime.now().strftime('%Y%m%d')}.log")

_FORMATTER = logging.Formatter(
    "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_logger(name:str , level:str = "INFO")->logging.Logger:
    """Return Configured logger for a module
    Usage in every file from src.core.logger import get_logger
    logger = get_logger(__name__)"""

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging,level.upper(),logging.INFO))

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(_FORMATTER)
    console.stream = open (
        sys.stdout.fileno(),
        mode="w",
        encoding="utf-8",
        buffering=1,
    )

    file_handler=logging.FileHandler(_LOG_FILE,encoding="utf-8")
    file_handler.setFormatter(_FORMATTER)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger


