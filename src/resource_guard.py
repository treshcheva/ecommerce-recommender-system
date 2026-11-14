"""Conservative sampled resource guard for sequential local experiments."""

import json
import os
import signal
import threading
import time

import psutil

GIB = 2 ** 30


def resource_snapshot():
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(os.getcwd())
    return {"ram_total_gib": memory.total / GIB,
            "ram_available_gib": memory.available / GIB,
            "swap_used_gib": psutil.swap_memory().used / GIB,
            "disk_free_gib": disk.free / GIB,
            "load_1m": psutil.getloadavg()[0], "logical_cpus": psutil.cpu_count()}


class ResourceGuard:
    """Log RSS of this runner + descendants; stop only owned child processes.

    RSS sums count shared pages more than once, so the bound is conservative.
    Peaks are sampled at 250 ms and are not exact OS high-water marks.
    """

    def __init__(self, log_path, max_rss_gib=1.25, min_available_gib=0.5,
                 max_swap_growth_gib=0.25, max_stage_seconds=1200):
        self.log_path = log_path
        self.limits = dict(max_rss_gib=max_rss_gib, min_available_gib=min_available_gib,
                           max_swap_growth_gib=max_swap_growth_gib,
                           max_stage_seconds=max_stage_seconds)
        self.initial = resource_snapshot()
        self.parent = psutil.Process()
        self.stop_event = threading.Event()
        self.stage = "preflight"
        self.stage_started = self.started = time.monotonic()
        self.reason = None
        self.peaks = {}
        self.thread = threading.Thread(target=self._watch, daemon=True)

    def set_stage(self, stage):
        self.check()
        self.stage, self.stage_started = stage, time.monotonic()

    def check(self):
        if self.reason:
            raise RuntimeError(self.reason)

    def __enter__(self):
        if self.initial['ram_available_gib'] < max(0.75, self.limits['min_available_gib'] + 0.5):
            raise RuntimeError("Insufficient available RAM before launch.")
        if self.initial['disk_free_gib'] < 5:
            raise RuntimeError("Less than 5 GiB free disk before launch.")
        if self.initial['load_1m'] > 2 * self.initial['logical_cpus']:
            raise RuntimeError("System load is too high before launch.")
        self.thread.start()
        return self

    def _watch(self):
        with self.log_path.open('a') as log:
            while not self.stop_event.is_set():
                try:
                    children = self.parent.children(recursive=True)
                    rss = self.parent.memory_info().rss
                    for child in children:
                        try:
                            rss += child.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                    sample = resource_snapshot()
                    sample.update(stage=self.stage, elapsed_seconds=time.monotonic() - self.started,
                                  stage_seconds=time.monotonic() - self.stage_started,
                                  process_tree_rss_gib=rss / GIB)
                    self.peaks[self.stage] = max(self.peaks.get(self.stage, 0), rss / GIB)
                    log.write(json.dumps(sample) + '\n')
                    log.flush()
                    if rss / GIB > self.limits['max_rss_gib']:
                        self.reason = "Runner + kernel RSS exceeded {:.2f} GiB".format(self.limits['max_rss_gib'])
                    elif sample['ram_available_gib'] < self.limits['min_available_gib']:
                        self.reason = "Available RAM fell below {:.2f} GiB".format(self.limits['min_available_gib'])
                    elif (self.limits['max_swap_growth_gib'] is not None
                          and sample['swap_used_gib'] - self.initial['swap_used_gib'] > self.limits['max_swap_growth_gib']):
                        self.reason = "Swap grew by more than {:.2f} GiB".format(self.limits['max_swap_growth_gib'])
                    elif sample['stage_seconds'] > self.limits['max_stage_seconds']:
                        self.reason = "Stage exceeded time limit"
                    elif sample['disk_free_gib'] < 3:
                        self.reason = "Free disk fell below 3 GiB"
                    if self.reason:
                        # The Jupyter client surfaces the interrupted stage as failure.
                        for child in reversed(children):
                            try:
                                child.send_signal(signal.SIGTERM)
                            except psutil.NoSuchProcess:
                                pass
                        break
                except Exception as error:
                    self.reason = "Resource monitoring failed: {}".format(error)
                    for child in self.parent.children(recursive=True):
                        try:
                            child.terminate()
                        except psutil.NoSuchProcess:
                            pass
                    break
                self.stop_event.wait(0.25)

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join(timeout=3)

    def summary(self):
        return {"initial": self.initial, "limits": self.limits,
                "sample_period_seconds": 0.25, "sampled_rss_peak_gib_by_stage": self.peaks,
                "stop_reason": self.reason,
                "rss_definition": "Sampled sum of runner and descendant RSS; includes shared-page double counting."}
