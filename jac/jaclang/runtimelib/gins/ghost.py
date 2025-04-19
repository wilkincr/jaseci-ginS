"""The Shell Ghost code for gins
"""

import os
import enum
import threading
import time
import logging
import types
import shutil 
import re
import json
from collections import defaultdict
from jaclang.runtimelib.gins.model import Gemini
from jaclang.runtimelib.gins.tracer import CFGTracker, CfgDeque
from pydantic import BaseModel

active_shell_ghost = None

# Helper class to maintain a fixed deque size
def read_source_code(file_path: str) -> str:
    with open(file_path, 'r') as f:
        return f.read()

class ShellGhost:
    def __init__(self, source_file_path: str | None = None):
        global active_shell_ghost
        active_shell_ghost = self
        self.cfgs = None
        self.cfg_cv = threading.Condition()
        self.tracker = CFGTracker()
        self.sem_ir = None
        self.machine = None

        self.finished_exception_lock = threading.Lock()
        self.exception = None
        self.finished = False
        self.variable_values = None

        self.model = Gemini()

        self.deque_lock = threading.Lock()
        self.__cfg_deque_dict = dict()
        self.__cfg_deque_size = 10
        self.source_file_path = os.path.abspath(source_file_path) if source_file_path else None

        self.logger = logging.getLogger()
        if self.logger.hasHandlers():
          self.logger.handlers.clear()
        logging.basicConfig(
        level=logging.INFO,             # Set the log level
        format='%(asctime)s - %(message)s', # Set the log message format
        datefmt='%Y-%m-%d %H:%M:%S',    # Set the timestamp format
          handlers=[
              logging.FileHandler("test.txt", mode='a'),  # Log to a file in append mode
          ]
        )
        

    def set_cfgs(self, cfgs):
        self.cfg_cv.acquire()
        self.cfgs = cfgs
        self.cfg_cv.notify()
        self.cfg_cv.release()

    def set_machine(self, machine):
        self.machine = machine

    def update_cfg_deque(self, cfg, module):
        self.deque_lock.acquire()
        if module not in self.__cfg_deque_dict:
          self.__cfg_deque_dict[module] = CfgDeque(self.__cfg_deque_size)
        self.__cfg_deque_dict[module].add_cfg(cfg)
        self.deque_lock.release()

    def get_cfg_deque_repr(self):
        return self.__cfg_deque.display_cfgs()

    def start_ghost(self):
        self.__ghost_thread = threading.Thread(target=self.worker)
        self.__ghost_thread.start()

    def set_finished(self, exception: Exception = None):
        self.finished_exception_lock.acquire()
        self.exception = exception
        self.finished = True
        self.finished_exception_lock.release()

    def assert_smart(self, condition: bool):
        from jaclang.runtimelib.gins.smart_assert import smart_assert
        smart_assert(condition, model=self.model)
    
    def worker_update_once(self):
        update_cfg()

    def _do_cfg_update(self):
        """
        One pass of tracer → CFG update.
        Mirrors the body of your old inline update_cfg closure.
        """
        exec_insts = self.tracker.get_exec_inst()
        if not exec_insts or not self.cfgs:
            return

        for module, offset_list in exec_insts.items():
            try:
                cfg = self.cfgs[module]

                # initialize per‐module state if needed
                if not hasattr(self, "_current_bb"):
                    self._current_bb, self._bb_entry_time = {}, {}
                if module not in self._current_bb:
                    self._current_bb[module]    = 0
                    cfg.block_map.idx_to_block[0].exec_count = 1
                    _, _, first_ts = offset_list[0]
                    self._bb_entry_time[module] = first_ts

                # walk through offsets and bump counts/times
                for offset, _, timestamp in offset_list:
                    curr = self._current_bb[module]
                    if offset not in cfg.block_map.idx_to_block[curr].bytecode_offsets:
                        for nxt in cfg.edges[curr]:
                            if offset in cfg.block_map.idx_to_block[nxt].bytecode_offsets:
                                elapsed = timestamp - self._bb_entry_time[module]
                                cfg.block_map.idx_to_block[curr].total_time += elapsed
                                cfg.edge_counts[(curr, nxt)] += 1
                                cfg.block_map.idx_to_block[nxt].exec_count += 1
                                self._bb_entry_time[module] = timestamp
                                self._current_bb[module]    = nxt
                                break
                # push into your deque and log
                self.update_cfg_deque(cfg.get_cfg_repr(), module)
                self.logger.info(cfg.to_json())
            except Exception as e:
                self.logger.error(f"Error updating CFG for {module}: {e}")

        # refresh any tracked variables
        self.variable_values = self.tracker.get_variable_values()

    def annotate_source_code(self):        
        source_code = read_source_code(self.source_file_path)

        bb_line_map = defaultdict(set)    # line_num -> [bb0]
        bb_jumps = defaultdict(list)      # bb0 -> [bb1, bb2]
        instr_line_map = defaultdict(list)
        
        bb_runtime_map = {}   # bb_id -> (exec_count, total_time)
        bb_memory_map = {}    # bb_id -> memory_usage
        
        mem_data = self.tracker.get_memory_usage()
        if mem_data:
            for module, usage_list in mem_data.items():
                if module in self.cfgs:
                    cfg = self.cfgs[module]
                    for offset, line_no, top_stats in usage_list:
                        for block_id, block in cfg.block_map.idx_to_block.items():
                            if any(offset in block.bytecode_offsets for offset in block.bytecode_offsets):
                                if block_id not in bb_memory_map:
                                    bb_memory_map[block_id] = 0
                                bb_memory_map[block_id] += sum(stat.size_diff for stat in top_stats)

        for module, cfg in self.cfgs.items():
            current_bb = None
            last_valid_lineno = None

            for block_id, block in cfg.block_map.idx_to_block.items():
                bb_runtime_map[block_id] = (block.exec_count, block.total_time)
            for line in cfg.display_instructions().splitlines():
                if match := re.search(r'^(?:Node )?(bb\d+):', line):
                    current_bb = match.group(1)

                elif "Instr:" in line and current_bb:
                    m = re.search(r'Lineno=(\d+)', line)
                    if m:
                        lineno = int(m.group(1))
                        last_valid_lineno = lineno
                    elif last_valid_lineno is not None:
                        lineno = last_valid_lineno
                    else:
                        continue  

                    bb_line_map[lineno].add(current_bb)

                    opname = re.search(r'Opname=([^,]+)', line)
                    arg = re.search(r'arg=(\S+)', line)
                    argval = re.search(r'argval=([^,]+)', line)
                    argrep = re.search(r'argrepr=([^,]+)', line)
                    jump_t = re.search(r'jump_t=(\w+)', line)

                    opname_str = opname.group(1) if opname else "UNKNOWN"
                    summary = f"{opname_str}"
                    if argrep and argrep.group(1):
                        summary += f" {argrep.group(1)}"
                    elif argval and argval.group(1) != 'None':
                        summary += f" {argval.group(1)}"
                    if jump_t:
                        summary += f" [jump_t={jump_t.group(1)}]"
                    if arg and arg.group(1) not in ['None', '']:
                        summary += f" [arg={arg.group(1)}]"

                    instr_line_map[lineno].append(summary)

            current_bb = None
            for line in cfg.get_cfg_repr().splitlines():
                if match := re.search(r'Node (bb\d+)', line):
                    current_bb = match.group(1)
                elif match := re.findall(r'-> (bb\d+)', line):
                    bb_jumps[current_bb].extend(match)

        annotated_lines = []
        for idx, line in enumerate(source_code.splitlines(), start=1):
            bb_comment = ""
            if idx in bb_line_map:
                parts = []
                for bb in bb_line_map[idx]:
                    jumps = bb_jumps.get(bb, [])
                    bb_id = int(bb[2:])
                    
                    runtime_info = ""
                    if bb_id in bb_runtime_map:
                        exec_count, total_time = bb_runtime_map[bb_id]
                        runtime_info = f" [exec={exec_count}, time={total_time:.4f}s]"
                    
                    memory_info = ""
                    if bb_id in bb_memory_map:
                        memory_usage = bb_memory_map[bb_id]
                        memory_info = f" [mem={memory_usage} bytes]"
                    
                    if jumps:
                        parts.append(f"{bb} → {', '.join(jumps)}{runtime_info}{memory_info}")
                    else:
                        parts.append(f"{bb}{runtime_info}{memory_info}")
                bb_comment = "  # " + " | ".join(parts)

            full_line = line + bb_comment
            annotated_lines.append(full_line)

            # Uncomment to add instruction information
            # 2. Add single-line instruction array annotation
            # if idx in instr_line_map:
            #     instr_group = ", ".join(instr_line_map[idx])
            #     annotated_lines.append(" " * 8 + f"#   [{instr_group}]")

        annotated_code = "\n".join(annotated_lines)
        print(annotated_code)
        
        # Save to file
        output_path = self.source_file_path + ".annotated.jac"
        with open(output_path, "w") as f:
            f.write(annotated_code)
        print(f"\n Annotated source written to: {output_path}")
        
        return annotated_code
    
    def write_error_response_to_file(self, error_response, output_filename=None):
        
        if self.source_file_path:
            source_dir = os.path.dirname(self.source_file_path)
            base_name = os.path.basename(self.source_file_path)
            source_name = os.path.splitext(base_name)[0]
        else:
            source_dir = "."  
            source_name = "unknown_source"
        
        if not output_filename:
            timestamp = int(time.time())
            filename = f"{source_name}_error_analysis_{timestamp}.json"
        else:
            filename = output_filename
        
        if not filename.endswith('.json'):
            filename += '.json'
        
        full_path = os.path.join(source_dir, filename)
        
        try:
            json_str = json.dumps(error_response, indent=4)
            
            with open(full_path, 'w') as f:
                f.write(json_str)
            
            print(f"Error analysis written to {full_path}")
            return True
        except Exception as e:
            print(f"Error writing to file: {e}")
            return False
    def prompt_annotated_code(self, annotated_code):
        prompt = f"""
        I have a JacLang program annotated with control flow and bytecode information.
        Each line may include:
        - Basic block transitions (e.g., `# bb0 → bb1, bb2`)
        - Bytecode instructions generated from that line (e.g., `#   [LOAD_NAME n, BINARY_OP /]`)

        Please analyze the code carefully and identify:
        1. Any **runtime errors** that may occur (e.g., division by zero, calling null values, type mismatches)
        2. **Logic bugs or unreachable code paths** based on the control flow
        3. **Control flow oddities**, such as:
        - basic blocks with no successors
        - jumps that never happen
        4. Opportunities for **safety improvements**, such as:
        - validating inputs before use
        - guarding against dangerous instructions like division, jumps, or external calls
        5. Any **performance improvements** or simplifications
        6. Point out which **line numbers** the issue is in

        Here is the annotated code:
        {annotated_code}
        """

        error_response = self.model.generate_structured(prompt)
        self.write_error_response_to_file(error_response)
        print(error_response)
        response = ""
        for improvement in error_response['improvement_list']:
            prompt = f"""
            I have a JacLang program annotated with control flow and bytecode information.
            Each line may include:
            - Basic block transitions (e.g., `# bb0 → bb1, bb2`)
            - Bytecode instructions generated from that line (e.g., `#   [LOAD_NAME n, BINARY_OP /]`)
            Here is the annotated code:
            {annotated_code}
            This code can be improved as follows:
            {improvement}
            Please provide ONLY actual executable code that fixes this error.
            The code should handle the error case properly following these requirements:
            1. Don't change the overall behavior of the program
            2. Add appropriate safety checks to prevent the error
            3. Return only the code that needs to be injected, no explanations
            
            Format your response as a Python code block starting with ```python and ending with ```\
            """
            response = self.model.generate_fixed_code(prompt)

            lines = response["fix_code"].splitlines()
            response["fix_code"] = "\n".join(line for line in lines if not line.strip().startswith("```"))
            print(prompt)
            print(response)
        return response
    


    def apply_fix_to_source(self, fix_code, start_line, end_line=None):
        with open(self.source_file_path, 'r') as f:
            lines = f.readlines()
    
        backup_path = f"{self.source_file_path}.backup.{int(time.time())}"
        shutil.copy2(self.source_file_path, backup_path)
        print(f"Created backup at {backup_path}")
        
        if end_line is None:
            end_line = start_line
        
        if 1 <= start_line <= len(lines) and 1 <= end_line <= len(lines) and start_line <= end_line:
            current_line = lines[start_line-1]
            indentation = ''
            for char in current_line:
                if char in [' ', '\t']:
                    indentation += char
                else:
                    break
            
            fix_lines = fix_code.strip().split('\n')
            indented_fix_lines = []
            
            for line in fix_lines:
                if line.strip():
                    indented_fix_lines.append(indentation + line + '\n')
                else:
                    indented_fix_lines.append('\n')
            
            lines[start_line-1:end_line] = indented_fix_lines
            
            with open(self.source_file_path, 'w') as f:
                f.writelines(lines)
            
            if start_line == end_line:
                print(f"Applied fix to line {start_line} in {self.source_file_path}")
            else:
                print(f"Applied fix to lines {start_line}-{end_line} in {self.source_file_path}")
            return True
        else:
            print(f"Error: Line range {start_line}-{end_line} is invalid (file has {len(lines)} lines)")
            return False

    def apply_multiple_fix_up(self):
        annotated_code = self.annotate_source_code()
        prompt = f"""
        I have a JacLang program annotated with control flow and bytecode information.
        Each line may include:
        - Basic block transitions (e.g., `# bb0 → bb1, bb2`)
        - Bytecode instructions generated from that line (e.g., `#   [LOAD_NAME n, BINARY_OP /]`)

        Please analyze the code carefully and identify:
        1. Any **runtime errors** that may occur (e.g., division by zero, calling null values, type mismatches)
        2. **Logic bugs or unreachable code paths** based on the control flow
        3. **Control flow oddities**, such as:
        - basic blocks with no successors
        - jumps that never happen
        4. Opportunities for **safety improvements**, such as:
        - validating inputs before use
        - guarding against dangerous instructions like division, jumps, or external calls
        5. Any **performance improvements** or simplifications
        6. Point out which **line numbers** the issue is in

        Here is the annotated code:
        {annotated_code}"""

        error_response = self.model.generate_structured(prompt)

        response = ""
        for improvement in error_response['improvement_list']:
            prompt = f"""
            I have a JacLang program annotated with control flow and bytecode information.
            Each line may include:
            - Basic block transitions (e.g., `# bb0 → bb1, bb2`)
            - Bytecode instructions generated from that line (e.g., `#   [LOAD_NAME n, BINARY_OP /]`)
            Here is the annotated code:
            {annotated_code}
            This code can be improved as follows:
            {improvement}
            Please provide ONLY actual executable code that fixes this error.
            The code should handle the error case properly following these requirements:
            1. Don't change the overall behavior of the program
            2. Add appropriate safety checks to prevent the error
            3. Return only the code that needs to be injected, no explanations
            
            Format your response as a Python code block starting with ```python and ending with ```\
            """
            print(improvement)
            response = self.model.generate_fixed_code(prompt)

            lines = response["fix_code"].splitlines()
            response["fix_code"] = "\n".join(line for line in lines if not line.strip().startswith("```"))
            self.apply_fix_to_source(start_line=response["start_line"], end_line=response["end_line"], fix_code=response["fix_code"])
            annotated_code = self.annotate_source_code()
            print(prompt)
            print(response)
        return response


    def worker(self):
        self.cfg_cv.acquire()
        if self.cfgs == None:
            print("waiting")
            self.cfg_cv.wait()

        self.cfg_cv.release()
        current_executing_bbs = {}
        bb_entry_time = {}
        
        # def update_cfg():
        #     exec_insts = self.tracker.get_exec_inst()

        #     if not exec_insts:
        #         return

        #     for module, offset_list in exec_insts.items():
        #         try:
        #             cfg = self.cfgs[module]

        #             if module not in current_executing_bbs:
        #                 current_executing_bbs[module] = 0
        #                 cfg.block_map.idx_to_block[0].exec_count = 1
        #                 _, _, first_time = offset_list[0]
        #                 bb_entry_time[module] = first_time

        #             for offset_tuple in offset_list:
        #                 offset = offset_tuple[0]
        #                 timestamp = offset_tuple[2]
        #                 current_bb = current_executing_bbs[module]

        #                 if offset not in cfg.block_map.idx_to_block[current_executing_bbs[module]].bytecode_offsets:
        #                     for next_bb in cfg.edges[current_executing_bbs[module]]:
        #                         if offset in cfg.block_map.idx_to_block[next_bb].bytecode_offsets:
        #                             if module in bb_entry_time:
        #                                 elapsed = timestamp - bb_entry_time[module]
        #                                 cfg.block_map.idx_to_block[current_bb].total_time += elapsed
        #                             else:
        #                                 elapsed = 0
        #                             bb_entry_time[module] = timestamp

        #                             cfg.edge_counts[(current_executing_bbs[module], next_bb)] += 1
        #                             cfg.block_map.idx_to_block[next_bb].exec_count += 1
        #                             current_executing_bbs[module] = next_bb
        #                             break

        #                 assert offset in cfg.block_map.idx_to_block[current_executing_bbs[module]].bytecode_offsets

        #         except Exception as e:

        #             print(f"Exception found: {e}")
                    
        #             #annotated_code = self.annotate_source_code()
                    
        #             # fix_code = self.prompt_annotated_code(annotated_code)
        #             # if fix_code:
        #             #     self.apply_fix_to_source(fix_code["line_number"], fix_code["fix_code"])
        #             #     continue
        #             # else:
        #             #     self.set_finished(e)
        #             #     print(e)
        #             #     break

        #     self.variable_values = self.tracker.get_variable_values()
        #     # print("Updating cfg deque")
        #     self.update_cfg_deque(cfg.get_cfg_repr(), module)
        #     self.logger.info(cfg.to_json())
        #     # print(f"CURRENT INPUTS: {self.tracker.get_inputs()}")

        self.finished_exception_lock.acquire()
        while not self.finished:
            self.finished_exception_lock.release()

            time.sleep(3)
            print("\nUpdating cfgs")
            self._do_cfg_update()
            self.finished_exception_lock.acquire()

        self.finished_exception_lock.release()

        print("\nUpdating cfgs at the end")
        self._do_cfg_update()
        #Multiple fixes needed
        # self.apply_multiple_fix_up()

        # One fix only 
        #annotated_code = self.annotate_source_code()
        #fix_code = self.prompt_annotated_code(annotated_code)
        # if fix_code:
        #     self.apply_fix_to_source(start_line=fix_code["start_line"], end_line=fix_code["end_line"], fix_code=fix_code["fix_code"])
    def worker_update_once(self):
        """
        Public hook: do exactly one CFG/tracer → CFG update
        so exec_count and total_time are fresh.
        """
        # ensure state dicts exist
        if not hasattr(self, "_current_bb"):
            self._current_bb, self._bb_entry_time = {}, {}
        self._do_cfg_update()

   