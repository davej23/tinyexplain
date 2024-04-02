from os import getenv

import logging


logging.basicConfig(level=getenv("TELOGLEVEL", "INFO"), format="%(asctime)-10s [ %(levelname)-8s ]  %(message)s")


class Logger:
    _logger: logging.Logger = logging.getLogger("tinyexplain")

    @classmethod
    def info(cls, value: str) -> None:
        cls._logger.info(value)

    @classmethod
    def debug(cls, value: str) -> None:
        cls._logger.debug(value)
