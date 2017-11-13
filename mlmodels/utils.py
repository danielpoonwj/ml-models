from functools import wraps
from time import time


def camel_to_snake(input_text):
    temp_list = []
    for idx, c in enumerate(input_text):
        if c.isupper() and idx > 0:
            temp_list.append('_')
        temp_list.append(c)
    return ''.join(temp_list).lower()


def time_taken(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time()

        result = f(*args, **kwargs)

        m, s = divmod(time() - start_time, 60)
        print('Time taken: %0.0f Minutes %0.0f Seconds' % (m, s))

        return result
    return wrapper
