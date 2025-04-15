import types
from typing import Optional, Callable
import threading
import signal
import sys
import copy
import os
import dis
from collections import deque, defaultdict



class CfgDeque:
    def __init__(self, max_size: int = 10):
        self.__max_size = max_size
        self.__deque = deque()

    def add_cfg(self, cfg_repr: str):
        self.__deque.append(cfg_repr)
        if len(self.__deque) > self.__max_size:
            self.__deque.popleft()

    def get_latest_cfg(self):
        return self.__deque[-1]
        
    def __len__(self):
        return len(self.__deque)

    def get_cfg_repr(self):
        res = [f"CFG Changes in last {len(self.__deque)} Updates:\n"]
        for idx, cfg in enumerate(self.__deque):
            res.append(f"\nCFG {idx+1} of {len(self.__deque)}\n")
            res.append(cfg)
        return "".join(res)


class CFGTracker:
    def __init__(self):
       
        self.curr_variables_lock = threading.Lock()
        self.curr_variables = {}
        self.line_freq = defaultdict(int)
        self.var_value_freq = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))


        # tracking inputs
        self.inputs = []

        # ========== NEW memory usage stuff ==========
        self.memory_usage = {}  # dict[module, list of (basic block, memory usage)]
        self.mem_lock = threading.Lock()

        self._prev_snapshot = None
        self._opcode_count = 0


    PRIMITIVES = (int, float, str, bool, type(None))
    SKIP_VARS = {"__name__", "__doc__", "__package__", "__loader__", "__spec__", "__file__"}


    def cpu_sample_handler(self, signum, frame):
        print("CPU Sample")
        if frame is None:
            return

        fname = frame.f_code.co_name
        lineno = frame.f_lineno
        print(lineno)
        self.line_freq[lineno] += 1
        print(f"{fname}:{lineno}")
        local_vars = frame.f_locals
        for name, value in local_vars.items():
            if name in self.SKIP_VARS:
                continue
            if isinstance(value, self.PRIMITIVES):
                var_key = (fname, lineno)
                self.var_value_freq[var_key][name][value] += 1


    def start_sampling(self, interval_sec=0.1):
        print("Sending signal")
        signal.signal(signal.SIGALRM, self.cpu_sample_handler)
        signal.setitimer(signal.ITIMER_REAL, interval_sec, interval_sec)
    
    def stop_sampling(self):
        signal.setitimer(signal.ITIMER_REAL, 0)

    
    def get_runtime_info(self, top_k=5):
        runtime_info = {}
        for (fname, lineno), var_dict in self.var_value_freq.items():
            var_summary = []
            for var_name, freq_dict in var_dict.items():
                top_k = sorted(freq_dict.items(), key=lambda x: -x[1])[:5]
                var_summary.append((var_name, top_k))
                runtime_info[(fname, lineno)] = (self.line_freq[lineno], var_summary)
        
        print(runtime_info)
        return runtime_info


    # @staticmethod
    # def get_line_from_frame(frame):
    #     lineno = frame.f_code.co_firstlineno
    #     byte_offset = frame.f_lasti
    #     line_starts = list(dis.findlinestarts(frame.f_code))
    #     current_line = lineno
    #     for offset, line in line_starts:
    #         if byte_offset < offset:
    #             break
    #         current_line = line
    #     return current_line