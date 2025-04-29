import types
from typing import Optional, Callable
import threading
import sys
import copy
import os
import traceback
import linecache
import dis
from collections import deque, defaultdict
import inspect
import warnings
import ast
import time 

import tracemalloc


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
        self.executed_insts = {}
        self.inst_lock = threading.Lock()

        self.curr_variables_lock = threading.Lock()
        self.curr_line = -1
        self.curr_variables = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(int)
                )
            )
        )    

        # tracking inputs
        self.inputs = []

        # ========== NEW memory usage stuff ==========
        self.memory_usage = {}  # dict[module, list of (basic block, memory usage)]
        self.mem_lock = threading.Lock()

        self._prev_snapshot = None
        self._opcode_count = 0

    def start_tracking(self):
        """Start tracking branch coverage and memory."""
        # Start tracemalloc
        tracemalloc.start()
        self._prev_snapshot = tracemalloc.take_snapshot()

        frame = sys._getframe()
        frame.f_trace_opcodes = True
        sys.settrace(self.trace_callback)

    def stop_tracking(self):
        """Stop tracking branch coverage and memory usage."""
        sys.settrace(None)
        # Optionally stop tracemalloc if you want
        # tracemalloc.stop()

    def get_exec_inst(self):
        with self.inst_lock:
            cpy = copy.deepcopy(self.executed_insts)
            self.executed_insts = {}
        return cpy

    def get_inputs(self):
        with self.inst_lock:
            cpy = copy.deepcopy(self.inputs)
            self.inputs = []
        return cpy
        
    def get_variable_values(self):
        with self.curr_variables_lock:
            cpy = copy.deepcopy(self.curr_variables)
        return cpy

    def get_memory_usage(self):
        """Returns and clears stored memory usage stats."""
        with self.mem_lock:
            cpy = copy.deepcopy(self.memory_usage)
            self.memory_usage = {}
        return cpy

    @staticmethod
    def get_line_from_frame(frame):
        lineno = frame.f_code.co_firstlineno
        byte_offset = frame.f_lasti
        line_starts = list(dis.findlinestarts(frame.f_code))
        current_line = lineno
        for offset, line in line_starts:
            if byte_offset < offset:
                break
            current_line = line
        return current_line

    def trace_callback(
        self, frame: types.FrameType, event: str, arg: any
    ) -> Optional[Callable]:
        code = frame.f_code
        # filter out irrelevant files
        if ".jac" not in code.co_filename:
            return self.trace_callback

        if event == "call":
            frame.f_trace_opcodes = True

        elif event == "opcode":
            filename = os.path.basename(code.co_filename)
            module = (
                code.co_name
                if code.co_name != "<module>"
                else os.path.splitext(filename)[0]
            )

            # 1) Record the executed instruction
            with self.inst_lock:
                if module not in self.executed_insts:
                    self.executed_insts[module] = []
                line_no = CFGTracker.get_line_from_frame(frame)
                self.executed_insts[module].append((frame.f_lasti, line_no, time.time()))
                
            # 2) Track variables
            if "__annotations__" in frame.f_locals:
                with self.curr_variables_lock:
                    lineno = frame.f_lineno
                    if lineno != self.curr_line and lineno is not None:
                        for var_name in frame.f_locals["__annotations__"]:
                            # Make sure the variable exists in local scope
                            if var_name in frame.f_locals:
                                variable_value = frame.f_locals[var_name]
                                
                                # Handle input tracking
                                if var_name == "input_val":
                                    current_input = copy.deepcopy(variable_value)
                                    if not self.inputs or current_input != self.inputs[-1]:
                                        self.inputs.append(current_input)
                                
                                # Store the value as a string representation
                                value_str = repr(variable_value)
                                
                                # Track the frequency using the string representation
                                self.curr_variables[module][lineno][var_name][value_str] += 1
                        
                        # Update current line after processing all variables
                        self.curr_line = lineno


            # 3) [NEW] Memory usage with tracemalloc
            #    - Here we do it for every opcode, but for performance you might do every Nth
            self._opcode_count += 1
            # e.g. only snapshot every Nth instructions to reduce overhead
            n = 100
            if self._opcode_count % n == 0:
                snapshot = tracemalloc.take_snapshot()
                if self._prev_snapshot is not None:
                    stats = snapshot.compare_to(self._prev_snapshot, "lineno")
                    top_stats = stats[:5]  # top 5 changes
                    with self.mem_lock:
                        if module not in self.memory_usage:
                            self.memory_usage[module] = []
                        # Store offset, line_no, plus the top stats for that range
                        self.memory_usage[module].append((frame.f_lasti, line_no, top_stats))
                self._prev_snapshot = snapshot

        return self.trace_callback