from functools import wraps
import time
import subprocess

from loguru import logger

PIPE = subprocess.PIPE
Popen = subprocess.Popen

@wraps(subprocess.run)
def run(cmd, *args, **kwargs):
    if isinstance(cmd, str):
        cmd = [cmd]
        kwargs['shell'] = True
    logger.info('{}', cmd)
    x = cmd[0]
    t0 = time.time()
    if not 'check' in kwargs:
        kwargs['check'] = True
    try:
        ret = subprocess.run(cmd, *args, **kwargs)
        return ret
    finally:
        t = time.time() - t0
        logger.info('{} took {}', x, t)

@wraps(subprocess.check_output)
def check_output(cmd, *args, **kwargs):
    x = cmd[0]
    t0 = time.time()
    try:
        ret = subprocess.check_output(cmd, *args, **kwargs)
        return ret
    finally:
        t = time.time() - t0
        logger.info('{} took {}', x, t)
