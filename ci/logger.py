#!/usr/bin/env python3

import os
import logging
from logging import Logger
from logging.handlers import TimedRotatingFileHandler
from logging import LogRecord
import re
import platform

container: str | None = os.environ.get("CONTAINER")
meta_tag: str | None = os.environ.get("META_TAG")
if container and meta_tag:
    dir: str = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "output", container, meta_tag)
    os.makedirs(dir, exist_ok=True)
    log_dir: str = os.path.join(dir, 'ci.log')
else:
    log_dir = os.path.join(os.getcwd(), 'ci.log')

logger: Logger = logging.getLogger()


class ColorPercentStyle(logging.PercentStyle):
    """Custom log formatter that add color to specific log levels."""
    grey: str = "38"
    yellow: str = "33"
    red: str = "31"
    cyan: str = "36"

    def _get_color_fmt(self, color_code, bold=False) -> str:
        if bold:
            return "\x1b[" + color_code + ";1m" + self._fmt + "\x1b[0m"
        return "\x1b[" + color_code + ";20m" + self._fmt + "\x1b[0m"

    def _get_fmt(self, levelno) -> str:
        colors: dict[int, str] = {
            logging.DEBUG: self._get_color_fmt(self.grey),
            logging.INFO: self._get_color_fmt(self.cyan),
            logging.WARNING: self._get_color_fmt(self.yellow),
            logging.ERROR: self._get_color_fmt(self.red),
            logging.CRITICAL: self._get_color_fmt(self.red)
        }

        return colors.get(levelno, self._get_color_fmt(self.grey))

    def _format(self, record: LogRecord) -> str:
        return self._get_fmt(record.levelno) % record.__dict__


class CustomLogFormatter(logging.Formatter):
    """Formatter that removes creds from logs."""
    ACCESS_KEY: str = os.environ.get("ACCESS_KEY", "super_secret_key")
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "super_secret_key")

    def formatException(self, exc_info) -> str:
        """Format an exception so that it prints on a single line."""
        result: str = super(CustomLogFormatter, self).formatException(exc_info)
        return repr(result)  # or format into one line however you want to

    def format_credential_key(self, s) -> str:
        return re.sub(self.ACCESS_KEY, '(removed)', s)

    def format_secret_key(self, s) -> str:
        return re.sub(self.SECRET_KEY, '(removed)', s)

    def format(self, record) -> str:
        s: str = super(CustomLogFormatter, self).format(record)
        if record.exc_text:
            s = s.replace('\n', '') + '|'
        s = self.format_credential_key(s)
        s = self.format_secret_key(s)

        return s

    def formatMessage(self, record) -> str:
        return ColorPercentStyle(self._fmt).format(record)


def configure_logging(log_level: str) -> None:
    """Setup console and file logging"""

    logger.handlers = []
    logger.setLevel(log_level)

    # Console logging
    ch = logging.StreamHandler()
    cf = CustomLogFormatter(
        '%(asctime)-15s | %(threadName)-17s | %(name)-10s | %(levelname)-8s | (%(module)s.%(funcName)s|line:%(lineno)d) | %(message)s |', '%d/%m/%Y %H:%M:%S')
    ch.setFormatter(cf)
    ch.setLevel(log_level)
    logger.addHandler(ch)

    # File logging
    fh = TimedRotatingFileHandler(
        log_dir, when="midnight", interval=1, backupCount=7, delay=True, encoding='utf-8')
    f = CustomLogFormatter(
        '%(asctime)-15s | %(threadName)-17s | %(name)-10s | %(levelname)-8s | (%(module)s.%(funcName)s|line:%(lineno)d) | %(message)s |', '%d/%m/%Y %H:%M:%S')
    fh.setFormatter(f)
    fh.setLevel(log_level)
    logger.addHandler(fh)

    logging.info('Operating system: %s', platform.platform())
    logging.info('Python version: %s', platform.python_version())
    if log_level.upper() == "DEBUG":
        logging.getLogger("botocore").setLevel(
            logging.WARNING)  # Mute boto3 logging output
        logging.getLogger("urllib3.connectionpool").setLevel(
            logging.WARNING)  # Mute urllib3.connectionpool logging output
