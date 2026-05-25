"""
Logging setup for the beatbox-midi project.

Call ``setup_logging()`` once at the process entry point 

Every other module then just does:

    import logging
    logger = logging.getLogger(__name__)

Because all modules live under the ``training`` package, their loggers are children of the ``training`` logger configured here and inherit its handlers automatically.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# Third-party libraries that flood the log with low-signal messages.
_SILENT_LOGGERS = [
    "datasets",
    "huggingface_hub",
    "urllib3",
    "filelock",
    "fsspec",
]

_CONSOLE_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%H:%M:%S" 
_FILE_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_FILE_DATE = "%Y-%m-%d %H:%M:%S" # full timestamp in the log file


def setup_logging(console_level: int = logging.INFO) -> logging.Logger:
    """
    Configure the ``training`` logger with a console and a rotating file handler.

    Creates a timestamped ``.log`` file under ``logs/`` in the project root.
    - The file handler captures everything at DEBUG and above
    - The console handler is limited to *console_level* (default: INFO) to avoid noise.

    Args:
        console_level: Minimum log level shown in the terminal.

    Returns:
        The configured ``training`` logger.
    """
    logs_dir = Path(__file__).resolve().parents[2] / "logs"
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = logs_dir / f"beatbox_{timestamp}.log"

    # Handlers 
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(
        logging.Formatter(_CONSOLE_FMT, datefmt=_DATE_FMT)
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # capture everything in the file
    file_handler.setFormatter(
        logging.Formatter(_FILE_FMT, datefmt=_FILE_DATE)
    )

    # training logger
    # Configuring the parent "training" logger is enough — all training.*
    # child loggers inherit both handlers automatically.
    logger = logging.getLogger("training")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:  # guard against duplicate setup on repeated calls
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    # Silence noisy third-party loggers 
    for name in _SILENT_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    logger.info("Logging to %s", log_file)
    return logger
