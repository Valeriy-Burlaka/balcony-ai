import time

from contextlib import contextmanager

from birds.lib.logger import get_logger


logger = get_logger("utils", verbosity=3)


@contextmanager
def timeit(task_name: str | None = None):
    start_time = time.monotonic()
    elapsed = {}
    try:
        yield elapsed
    finally:
        elapsed_time_seconds = round(time.monotonic() - start_time, 2)
        elapsed["seconds"] = elapsed_time_seconds
        if task_name:
            logger.debug(f"{task_name} took {elapsed_time_seconds} seconds.")
        else:
            logger.debug(f"It took {elapsed_time_seconds} seconds.")
