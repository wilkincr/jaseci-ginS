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
        
        # Create 5 different versions of annotations
        versions = {
            "source_only": list(source_code_lines),        # Just source code
            "with_bb": list(source_code_lines),            # With basic block info
            "with_instr": list(source_code_lines),         # With instructions
            "with_vars": list(source_code_lines),          # With variable values
            "complete": list(source_code_lines)            # With all annotations
        }
        base_output_path = self.source_file_path
        bb_runtime_map = {}   # bb_id -> (exec_count, total_time)
        instr_line_map = defaultdict(list)  # line_number -> [instruction strings]
        
        for module, cfg in self.cfgs.items():
            # Map basic blocks to their runtime metrics
            for block_id, block in cfg.block_map.idx_to_block.items():
                bb_runtime_map[block_id] = (block.exec_count, block.total_time)
                
                # Map instructions to their source lines
                for instr in block.instructions:
                    # If instruction has a line number, map it to that line
                    if instr.lineno is not None:
                        line_number = instr.lineno
                        instr_line_map[line_number].append(f"{instr.op}({instr.argval})")
                    # If no line number, map it to the line where the block starts
                    else:
                        # Find the first instruction in this block that has a line number
                        first_line = None
                        for first_instr in block.instructions:
                            if first_instr.lineno is not None:
                                first_line = first_instr.lineno
                                break
                        
                        # If we found a line number, use it
                        if first_line is not None:
                            instr_line_map[first_line].append(f"{instr.op}({instr.argval}) [no line]")
                        # If no instruction in this block has a line number, use the block_id as a key
                        else:
                            special_key = f"block_{block_id}"
                            if special_key not in instr_line_map:
                                instr_line_map[special_key] = []
                            instr_line_map[special_key].append(f"{instr.op}({instr.argval})")
            
            # Create the variable tracker to access variable information per block
            variable_tracker = VariableTracker(self.tracker.get_variable_values(),
                                            cfg.block_map.line_to_blocks)
            
            # Process each version with appropriate annotations
            for version_name in versions:
                current_bb = -1
                inserted_line_count = 0  # Track how many lines we've inserted for correct positioning
                
                # First, handle any special blocks with no line numbers
                special_blocks = [key for key in instr_line_map.keys() if isinstance(key, str) and key.startswith("block_")]
                if special_blocks and version_name in ["complete"]:
                    # Add a section at the top for instructions without line mappings
                    versions[version_name].insert(0, "# ==== INSTRUCTIONS WITHOUT LINE MAPPINGS ====")
                    inserted_line_count += 1
                    
                    for block_key in special_blocks:
                        block_id = int(block_key.split('_')[1])
                        execution_freq, execution_time = bb_runtime_map[block_id]
                        
                        block_header = f"# BB: {block_id} Execution frequency: {execution_freq} Total execution time: {execution_time:.3f} ms"
                        versions[version_name].insert(inserted_line_count, block_header)
                        inserted_line_count += 1
                        
                        instr_line = f"#   Instructions: [{', '.join(instr_line_map[block_key])}]"
                        versions[version_name].insert(inserted_line_count, instr_line)
                        inserted_line_count += 1
                        
                        # Add empty line for readability
                        versions[version_name].insert(inserted_line_count, "")
                        inserted_line_count += 1
                    
                    # Add separator after special blocks section
                    versions[version_name].insert(inserted_line_count, "# " + "=" * 50)
                    versions[version_name].insert(inserted_line_count + 1, "")
                    inserted_line_count += 2
            
                # Now process normal source lines
                var_printed_blocks = set()
                for line_number in range(1, len(source_code_lines) + 1):
                    adjusted_line_idx = line_number + inserted_line_count - 1
                    
                    # Add basic block annotations for appropriate versions
                    if line_number in cfg.block_map.line_to_blocks and version_name in ["with_bb", "complete"]:
                        block_id = cfg.block_map.line_to_blocks[line_number]
                        if block_id != current_bb:
                            current_bb = block_id
                            execution_freq, execution_time = bb_runtime_map[block_id]
                            versions[version_name][adjusted_line_idx] += f" # BB: {block_id} Execution frequency: {execution_freq} Total execution time: {execution_time:.3f} ms"
                        else:
                            versions[version_name][adjusted_line_idx] += f" # BB: {block_id}"
                    
                    # Add instruction annotations for appropriate versions
                    if line_number in instr_line_map and instr_line_map[line_number] and version_name in ["with_instr", "complete"]:
                        indent = len(versions[version_name][adjusted_line_idx]) - len(versions[version_name][adjusted_line_idx].lstrip())
                        instruction_line = " " * indent + f"#   Instructions: [{', '.join(instr_line_map[line_number])}]"
                        
                        # Insert the instruction annotation after the current line
                        versions[version_name].insert(adjusted_line_idx + 1, instruction_line)
                        inserted_line_count += 1
                    
                    # Add variable tracking information for appropriate versions
                    if line_number in cfg.block_map.line_to_blocks and version_name in ["with_vars", "complete"]:
                        block_id = cfg.block_map.line_to_blocks[line_number]
                        # Check if we have variable information for this module and block
                        if block_id not in var_printed_blocks:
                            var_dict = variable_tracker.variables.get(module, {}).get(block_id, {})
                            if var_dict:
                                # compute where to insert, indent, etc.
                                var_line_idx = adjusted_line_idx + 1
                                if version_name == "complete" and line_number in instr_line_map:
                                    var_line_idx += 1

                                indent = len(versions[version_name][var_line_idx - 1]) \
                                        - len(versions[version_name][var_line_idx - 1].lstrip())

                                parts = []
                                for var_name, val_counts in var_dict.items():
                                    top_vals = sorted(val_counts.items(), key=lambda x: -x[1])[:variable_tracker.top_k]
                                    for val, freq in top_vals:
                                        parts.append(f"{var_name} = {val} (seen {freq}×)")

                                var_line = " " * indent \
                                        + "#   Variable values in this block:   " \
                                        + "  ".join(parts)
                                versions[version_name].insert(var_line_idx, var_line)
                                inserted_line_count += 1

                                # mark done so we don’t print again
                                var_printed_blocks.add(block_id)
                                # Save all versions to appropriate files
            
            for version_name, annotated_lines in versions.items():
                annotated_code = "\n".join(annotated_lines)
                output_path = f"{base_output_path}.{version_name}.jac"
                
                with open(output_path, "w") as f:
                    f.write(annotated_code)
                
                print(f"\n {version_name} version written to: {output_path}")
                
                # Generate analysis for each version
                self.prompt_annotated_code(annotated_code, variable_tracker, version_name)
            
            return versions["complete"]  # Return the complete version for backward compatibility
    
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
        
    def prompt_annotated_code(self, annotated_code, variable_tracker, version_name="complete"):
        prompt = f"""
        I have a JacLang program annotated with control flow and bytecode information.
        Each line may include:
        """
        
        # Customize prompt based on the version
        if version_name == "source_only":
            prompt += """
            - No additional annotations, just source code
            """
        elif version_name == "with_bb":
            prompt += """
            - Basic block annotations (e.g., `# BB: 0 Execution frequency: 1 Total execution time: 0.001 ms`)
            """
        elif version_name == "with_instr":
            prompt += """
            - Bytecode instructions generated from that line (e.g., `#   Instructions: [SETUP_ANNOTATIONS(None)]`)
            """
        elif version_name == "with_vars":
            prompt += """
            - Variable values observed in this block (e.g., `#   Variable values in this block:   x = 0 (seen 3×)  y = 4 (seen 48×)`)
            """
        else:  
            prompt += """
            - Basic block transitions (e.g., `# BB: 0 Execution frequency: 1 Total execution time: 0.001 ms`)
            - Bytecode instructions generated from that line (e.g., `#   Instructions: [SETUP_ANNOTATIONS(None)]`)
            - Variable values observed in this block (e.g., `#   Variable values in this block:   x = 0 (seen 3×)  y = 4 (seen 48×)`)
            """
        
        prompt += f"""
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
        
        # Only add variable tracker information for versions that don't already include it
        if version_name not in ["with_vars", "complete"]:
            prompt += f"""
            Here is also the frequencies for dynamic values in variables in each basic block:
            {variable_tracker}
            """
        
        print(f"Generating analysis for {version_name} version")
        error_response = self.model.generate_structured(prompt)
        
        # Write the response to a file with a version-specific name
        output_filename = None
        if self.source_file_path:
            source_dir = os.path.dirname(self.source_file_path)
            base_name = os.path.basename(self.source_file_path)
            source_name = os.path.splitext(base_name)[0]
            timestamp = int(time.time())
            output_filename = f"{source_name}_error_analysis_{version_name}_{timestamp}.json"
        
        self.write_error_response_to_file(error_response, output_filename)
        print(error_response)
        return error_response
    

    
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
        
        # for module, cfg in self.cfgs.items():
        self.annotate_source_code()
            # variable_tracker = VariableTracker(self.tracker.get_variable_values(), cfg.block_map.line_to_blocks)



   