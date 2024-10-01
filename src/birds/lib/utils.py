import time

from contextlib import contextmanager


@contextmanager
def timer(task_name: str):
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    print(f"{task_name} took {elapsed_time:.2f} seconds.")
