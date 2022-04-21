"""
Takes information from the camera and returns interop output json file.
"""

import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor
import time

def wait_function(x, y):
    print('Task(', x,'multiply', y, ') started')
    print('Task(', x,'multiply', y, ') completed')
    return x * y

with ThreadPoolExecutor() as executor: #change max_workers to 2 and see the results
    future = executor.submit(wait_function, 3, 4)
    future2 = executor.submit(wait_function, 8, 8)
    future3 = executor.submit(wait_function, 12, 12)
    while True:
        if(future.running()):
            print("Task 1 running")
        if(future2.running()):
            print("Task 2 running")
        if(future3.running()):
            print("Task 3 running")

        if(future.done() and future2.done() and future3.done()):
            print(future.result(), future2.result()), future3.done()
            break
