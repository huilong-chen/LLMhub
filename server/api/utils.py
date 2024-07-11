import re
import uuid
import logging
import os

from logging.handlers import TimedRotatingFileHandler


def random_uuid() -> str:
    return str(uuid.uuid4().hex)

def get_api_key(authorization):
    if authorization:
        match = re.match("^Bearer (.+)$", authorization)
        if match:
            return match.group(1)
    return "unknown"

def setup_logger(log_dir, name=None):
    logger = logging.getLogger(name)

    logger.handlers = []
    logger.setLevel(logging.INFO)
    fmt, datefmt = _get_formatter(name)
    formatter = logging.Formatter(fmt, datefmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{name}.log"
    file_handler = TimedRotatingFileHandler(log_file, when="midnight", backupCount=15, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def _get_formatter(logger_name=""):
    datefmt = "%Y-%m-%d %H:%M:%S"
    name_format = f"[{logger_name}]" if logger_name else "[%(pathname)s:%(lineno)d]"
    fmt = [
        "%(asctime)-15s.%(msecs)03d",
        "[%(levelname)s]",
        "[%(processName)s:%(threadName)s]",
        name_format,
        "@@@traceId=N/A@@@",
        "%(message)s",
    ]
    fmt = " ".join(fmt)
    return fmt, datefmt