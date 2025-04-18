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
from jaclang.runtimelib.gins.variable_tracker import VariableTracker

# Helper class to maintain a fixed deque size
def read_source_code(file_path: str) -> str:
    with open(file_path, 'r') as f:
        return f.read()


class ShellGhost:
    def __init__(self, source_file_path: str | None = None):
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

    def annotate_source_code(self):        
        source_code = read_source_code(self.source_file_path)
        source_code_lines = source_code.splitlines()
        annotated_lines = list(source_code_lines)

        # bb_line_map = defaultdict(set)    # line_num -> [bb0]
        # bb_jumps = defaultdict(list)      # bb0 -> [bb1, bb2]
        # instr_line_map = defaultdict(list)
        
        bb_runtime_map = {}   # bb_id -> (exec_count, total_time)

        for module, cfg in self.cfgs.items():

            for block_id, block in cfg.block_map.idx_to_block.items():
                bb_runtime_map[block_id] = (block.exec_count, block.total_time)

            current_bb = -1
            for line_number in range(1, len(source_code_lines) + 1):
                if line_number in cfg.block_map.line_to_blocks:
                    block_id = cfg.block_map.line_to_blocks[line_number]
                    if block_id != current_bb:
                        current_bb = block_id
                        execution_freq, execution_time = bb_runtime_map[block_id]
                        # print("line_number", line_number, "block_at_line", block_at_line, " current_bb", current_bb)
                        # if block_at_line != current_bb:
                        #     current_bb += 1
                        annotated_lines[line_number - 1] += f" # BB: {block_id} Execution frequency: {execution_freq} Total execution time: {execution_time:.3f} ms"
                    else:
                        annotated_lines[line_number - 1] += f" # BB: {block_id}"

            
            # for line_number, block_id in cfg.block_map.line_to_blocks.items():
            #     print(line_number)
            #     annotated_lines[line_number] += f" # BB: {block_id}"


        # for idx, line in enumerate(source_code.splitlines(), start=1):
        #     bb_comment = ""
        #     bb_id = self.cfgs[module].block_map.block_for_line(idx)

        #     full_line = line + bb_comment
        #     annotated_lines.append(full_line)

        #     # Uncomment to add instruction information
        #     # 2. Add single-line instruction array annotation
        #     # if idx in instr_line_map:
        #     #     instr_group = ", ".join(instr_line_map[idx])
        #     #     annotated_lines.append(" " * 8 + f"#   [{instr_group}]")

        annotated_code = "\n".join(annotated_lines)        
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
    def prompt_annotated_code(self, annotated_code, variable_tracker):
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
        Here is also the frequencies for dynamic values in variables in each basic block:
        {variable_tracker}
        """
        print("prompt", prompt)

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
        
        def update_cfg():
            exec_insts = self.tracker.get_exec_inst()

            if not exec_insts:
                return

            for module, offset_list in exec_insts.items():
                try:
                    cfg = self.cfgs[module]

                    if module not in current_executing_bbs:
                        current_executing_bbs[module] = 0
                        cfg.block_map.idx_to_block[0].exec_count = 1
                        _, _, first_time = offset_list[0]
                        bb_entry_time[module] = first_time

                    for offset_tuple in offset_list:
                        offset = offset_tuple[0]
                        timestamp = offset_tuple[2]
                        current_bb = current_executing_bbs[module]

                        if offset not in cfg.block_map.idx_to_block[current_executing_bbs[module]].bytecode_offsets:
                            for next_bb in cfg.edges[current_executing_bbs[module]]:
                                if offset in cfg.block_map.idx_to_block[next_bb].bytecode_offsets:
                                    if module in bb_entry_time:
                                        elapsed = timestamp - bb_entry_time[module]
                                        cfg.block_map.idx_to_block[current_bb].total_time += elapsed
                                    else:
                                        elapsed = 0
                                    bb_entry_time[module] = timestamp

                                    cfg.edge_counts[(current_executing_bbs[module], next_bb)] += 1
                                    cfg.block_map.idx_to_block[next_bb].exec_count += 1
                                    current_executing_bbs[module] = next_bb
                                    break

                        assert offset in cfg.block_map.idx_to_block[current_executing_bbs[module]].bytecode_offsets

                except Exception as e:

                    print(f"Exception found: {e}")
                    
                    # annotated_code = self.annotate_source_code()
                    
                    # fix_code = self.prompt_annotated_code(annotated_code)
                    # if fix_code:
                    #     self.apply_fix_to_source(fix_code["line_number"], fix_code["fix_code"])
                    #     continue
                    self.set_finished(e)
                    print(e)
                    break
            
            # print("Updating cfg deque")
            self.update_cfg_deque(cfg.get_cfg_repr(), module)
            self.logger.info(cfg.to_json())
            # print(f"CURRENT INPUTS: {self.tracker.get_inputs()}")

        self.finished_exception_lock.acquire()
        while not self.finished:
            self.finished_exception_lock.release()

            time.sleep(3)
            print("\nUpdating cfgs")
            update_cfg()
            self.finished_exception_lock.acquire()

        self.finished_exception_lock.release()

        print("\nUpdating cfgs at the end")
        update_cfg()
        for cfg in self.cfgs.values():
            print(cfg.get_cfg_repr())
        
        # print("\nAnnotating source code")
        # annotated_code = self.annotate_source_code()
        # print(annotated_code)
        #Multiple fixes needed
        # self.apply_multiple_fix_up()

        # One fix only
        for module, cfg in self.cfgs.items():
            annotated_code = self.annotate_source_code()
            variable_tracker = VariableTracker(self.tracker.get_variable_values(), cfg.block_map.line_to_blocks)
            fix_code = self.prompt_annotated_code(annotated_code, variable_tracker)
            print(fix_code)
        # if fix_code:
        #     self.apply_fix_to_source(start_line=fix_code["start_line"], end_line=fix_code["end_line"], fix_code=fix_code["fix_code"])


   