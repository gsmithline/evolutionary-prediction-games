from contextlib import contextmanager
import time

@contextmanager
def track_progress(message):
    start_time = time.time()
    print(f'{message}...',end=' ')
    try:
        yield start_time
    finally:
        end_time = time.time()
        print(f'Done! ({end_time-start_time:.2g}s)')

