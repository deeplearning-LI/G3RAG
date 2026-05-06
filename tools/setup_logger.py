import sys
from pathlib import Path
from loguru import logger


def setup_logger(workdir: str, day: str, timestamp: str, log_level: str = "INFO"):
    """
    Setup logger using loguru with rotation and console output.

    Args:
        workdir: Base workspace directory where log file will be saved.
        day: Log day string (e.g., "2025-07-04").
        timestamp: Timestamp string (e.g., "123456").
        log_level: Log level (e.g., "INFO", "DEBUG").

    Returns:
        loguru.logger instance
    """
    log_file = Path(workdir) / f"logs_{day}_{timestamp}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>"
    )

    logger.add(
        log_file,
        level=log_level,
        rotation="10 MB",
        retention=5,
        encoding="utf-8",
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )

    return logger
