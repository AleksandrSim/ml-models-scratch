import time


def print_timer(start_time, stop_time, algorithm_running, metric_used, accuracy):
    print(
        f"\n\nRun {algorithm_running} with metric: {metric_used} in {stop_time - start_time:0.3f} seconds with "
        f"an accuracy of {accuracy}%\n"
    )


def timer():
    return time.perf_counter()
