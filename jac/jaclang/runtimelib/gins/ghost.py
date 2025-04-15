"""The Shell Ghost code for gins
"""

import os
import enum
import threading
import time
import logging
import psutil
import time
import types
import shutil 
import re
from collections import defaultdict
from jaclang.runtimelib.gins.model import Gemini
from jaclang.runtimelib.gins.tracer import CFGTracker, CfgDeque
from jaclang.runtimelib.gins.ghostwriter import GhostWriter

def read_source_code(file_path: str) -> str:
    with open(file_path, 'r') as f:
        return f.read()


class ShellGhost:
    def __init__(self, source_file_path: str | None = None):
        self.cfgs = None
        self.cfg_cv = threading.Condition()
        self.tracker = CFGTracker()
        self.sem_ir = None

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
            level=logging.INFO,           # Set the log level
            format='%(asctime)s - %(message)s', # Log message format
            datefmt='%Y-%m-%d %H:%M:%S',  # Timestamp format
            handlers=[
                logging.FileHandler("test.txt", mode='a'),  # Log to a file (append mode)
            ]
        )

    def set_cfgs(self, cfgs):
        self.cfg_cv.acquire()
        self.cfgs = cfgs
        self.cfg_cv.notify()
        self.cfg_cv.release()

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
        with self.finished_exception_lock:
            self.exception = exception
            self.finished = True

    def prompt_direct(self):
      script = """
      import:py from math { exp }
      import:py from time { sleep }
      # Data structure representing system configuration
      glob system_config: Dict[str, Union[int, str, float]] = {
          'base_load': 1000,    # Base power load in watts
          'min_duration': 10,   # Minimum valid duration in minutes
          'mode': 'active',
          'time_step': 0,       # Track progression of simulation
          'reference_delta': 200 # Reference power delta for normalization
      };

      # Function to generate declining power readings
      with entry {
          # Create gradually converging power readings
          base: float = system_config['base_load'];
          power_readings: list[float] = [];
          time_periods: list[int] = [];
          reference_power: float = base + 200;# Reference power for normalization

          # Generate 200 readings that gradually approach base_load
          for i in range(200) {
              # Power gradually approaches base_load (1000W)
              delta: float = 200.0 * exp(-0.5 * i);# Slower decay for better visualization
              current_power: float = base + delta;
              power_readings.append(current_power);

              # Time periods increase linearly
              time_periods.append(15 + i * 2);
          }

          # Initialize results storage

          efficiency_metrics: list = [];
          total_operational_time: int = 0;

          # Process each power reading with different execution paths
          for (idx, current_power) in enumerate(power_readings) {
              if system_config['mode'] != 'active' {
                  continue;
              }

              duration: int = time_periods[idx];
              if duration < system_config['min_duration'] {
                  continue;
              }

              # Track simulation progression

              system_config['time_step'] += 1;

              power_delta: float = current_power - system_config['base_load'];

              # Introduce different execution paths based on time_step
              if system_config['time_step'] > 50 {
                  diminishing_reference: float = power_delta * 2;  # Reference point approaches zero with power_delta
                  power_utilization: float = power_delta / diminishing_reference;  # Approaches 0.5, then unstable
              } else {
                  # Original calculation path for first 10 steps
                  power_utilization: float = power_delta / system_config['reference_delta'];
              }
              period_efficiency: float = power_utilization * (duration / max(time_periods)) * 100;

              efficiency_metrics.append(period_efficiency);
              total_operational_time += duration;

              # Print current state
              print(
                  f"Step {system_config['time_step']}: Power={current_power}W, " + f"Delta from base={current_power - system_config['base_load']}W"
              );
          }

          # Calculate final metrics if no error occurred

          average_efficiency: float = sum(efficiency_metrics) / len(efficiency_metrics) if efficiency_metrics else 0;
          operational_hours: float = total_operational_time / 60;
          print(
              f"System Analysis Complete - Efficiency: {average_efficiency}%"
          );

      }

      """
      prompt = f"""
      I have the following script:
      {script}

      Can you identity bottlneck optimizations or where the code can error out?"
      Reason about the program using the provided information, reason about the program itself and what improvements could be made.
      """

      response = self.model.generate(prompt)

      print("\nGin Analysis(With static info):\n", response)
      
    def prompt_llm(self, verbose: bool = False):
        prompt = """I have a program.
        {cfgs},
        Instructions per basic block:
        {instructions}
        """
        # Semantic and Type information from source code:
        # {sem_ir}"""

        cfg_string = ""
        ins_string = ""
        for module, cfg in self.cfgs.items():
            cfg_string += f"Module: {module}\n{cfg}"
            ins_string += f"Module: {module}\n{cfg.display_instructions()}"

        prompt = prompt.format(
            cfgs=cfg_string, instructions=ins_string, sem_ir=self.sem_ir.pp()
        )

        if self.variable_values != None:
            prompt += "\nCurrent variable values at the specified bytecode offset:"

            for module, var_map in self.variable_values.items():
                prompt += f"\nModule {module}: Offset: {var_map[0]}, Variables: {str(var_map[1])}"

        self.finished_exception_lock.acquire()

        if self.exception:
            prompt += f"\nException: {self.exception}"

        self.finished_exception_lock.release()

        # prompt += "\nCan you identity bottlneck optimizations or where the code can error out?"
        # prompt += "\n(Reason about the program using cfg, semantic and type information. Instead of saying what BB could be improved, reason about the program itself and what improvements could be made.)"
        # prompt += "\n If variable values are available, reason about at what point did a variable cause an issue"
        prompt += "\n Please use the following information fill in predicted_edges[freq] for each BB edge with something completely random"
        if verbose:
            print(prompt)

        response = self.model.generate(prompt)

        print("\nGin Analysis:\n", response)
        return response

    def prompt_llm_with_history(self, verbose: bool = False):
        prompt = """I have a program.
        Up to last {history_size} CFGs recorded:
        {cfgs},
        Instructions per basic block:
        {instructions}
        Semantic and Type information from source code:
        {sem_ir}"""

        cfg_string = ""
        ins_string = ""
        for module, cfg in self.cfgs.items():
            cfg_history = "None at this time"
            if module in self.__cfg_deque_dict:
              cfg_history = self.__cfg_deque_dict[module].get_cfg_repr()
            cfg_string += f"Module: {module}\n{cfg_history}"
            ins_string += f"Module: {module}\n{cfg.display_instructions()}"

        prompt = prompt.format(
            history_size=self.__cfg_deque_size,
            cfgs=cfg_string, 
            instructions=ins_string, 
            sem_ir=self.sem_ir.pp()
        )

        if self.variable_values != None:
            prompt += "\nCurrent variable values at the specified bytecode offset:"

            for module, var_map in self.variable_values.items():
                prompt += f"\nModule {module}: Offset: {var_map[0]}, Variables: {str(var_map[1])}"

        self.finished_exception_lock.acquire()

        if self.exception:
            prompt += f"\nException: {self.exception}"

        self.finished_exception_lock.release()

        prompt += "\nCan you identity bottlneck optimizations or where the code can error out?"
        prompt += "\n(Reason about the program using cfg history, semantic and type information. Users will not have access to BB information, so try to reason about the logic and frequencies of blocks instead.)"
        prompt += "\n Additionally, look for any cases where the hot path of the code appears to change at some point in the program"
        prompt += "\n If variable values are available, can you provide tracing information to help find the root cause of any issues?"

        if verbose:
            print(prompt)

        response = self.model.generate(prompt)


        return response

    def annotate_source_code(self):
        source_code = read_source_code(self.source_file_path)

        bb_line_map = defaultdict(set)    # line_num -> [bb0]
        bb_jumps = defaultdict(list)       # bb0 -> [bb1, bb2]
        instr_line_map = defaultdict(list) 

        for module, cfg in self.cfgs.items():
            current_bb = None

            last_valid_lineno = None

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



            # Extract jumps between blocks
            current_bb = None
            for line in cfg.get_cfg_repr().splitlines():
                if match := re.search(r'Node (bb\d+)', line):
                    current_bb = match.group(1)
                elif match := re.findall(r'-> (bb\d+)', line):
                    bb_jumps[current_bb].extend(match)

        # Output annotated source
        annotated_lines = []
        for idx, line in enumerate(source_code.splitlines(), start=1):
            # 1. Append original line + BB comment
            bb_comment = ""
            if idx in bb_line_map:
                parts = []
                for bb in bb_line_map[idx]:
                    jumps = bb_jumps.get(bb, [])
                    if jumps:
                        parts.append(f"{bb} → {', '.join(jumps)}")
                    else:
                        parts.append(f"{bb}")
                bb_comment = "  # " + " | ".join(parts)

            full_line = line + bb_comment
            annotated_lines.append(full_line)

            # 2. Add single-line instruction array annotation
            if idx in instr_line_map:
                instr_group = ", ".join(instr_line_map[idx])
                annotated_lines.append(" " * 8 + f"#   [{instr_group}]")

        annotated_code = "\n".join(annotated_lines)
        print(annotated_code)
        return annotated_code

        # Save to file
        output_path = self.source_file_path + ".annotated.jac"
        with open(output_path, "w") as f:
            f.write(annotated_code)
        print(f"\n✅ Annotated source written to: {output_path}")

    def prompt_annotated_code(self, annotated_code):
        prompt = """
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
        6. If possible: point out which **line numbers** or **blocks** the issue is in

        Here is the annotated code:
        """ + annotated_code
        response = self.model.generate_structured(prompt)
        print(response)
        return response
    


    def prompt_for_runtime(self, verbose: bool = False):
        prompt = """I have a program.
        Up to last {history_size} CFGs recorded:
        {cfgs},
        Instructions per basic block:
        {instructions}
        Semantic and Type information from source code:
        {sem_ir}"""

        prompt = """I have a program.
        Up to last {history_size} CFGs recorded:
        {cfgs},
        Semantic and Type information from source code:
        {sem_ir}"""

        cfg_string = ""
        ins_string = ""
        for module, cfg in self.cfgs.items():
            cfg_history = "None at this time"
            if module in self.__cfg_deque_dict:
              cfg_history = self.__cfg_deque_dict[module].get_cfg_repr()
            cfg_string += f"Module: {module}\n{cfg_history}"
            ins_string += f"Module: {module}\n{cfg.display_instructions()}"

        # prompt = prompt.format(
        #     history_size=self.__cfg_deque_size,
        #     cfgs=cfg_string, 
        #     instructions=ins_string, 
        #     sem_ir=self.sem_ir.pp()
        # )

        prompt = prompt.format(
            history_size=self.__cfg_deque_size,
            cfgs=cfg_string, 
            sem_ir=self.sem_ir.pp()
        )

        if self.variable_values != None:
            prompt += "\nCurrent variable values at the specified bytecode offset:"

            for module, var_map in self.variable_values.items():
                prompt += f"\nModule {module}: Offset: {var_map[0]}, Variables: {str(var_map[1])}"

        # prompt +=   """\n given this information, what is the program behavior? What type of runtime error can occurs?
        #             If there are any possible runtime errors, what line number is the instruction that causes the error?
        #             """

        prompt +=   """\n Given this execution behavior at runtime:
                    What optimizations can be done to improve runtime performance based on the current input?
                    Consider improvements in algorithms, data structures, or other perforamance related optimizations.
                    """

        if verbose:
            print(prompt)
                    
        print("PROMPT: ", prompt)
        response = self.model.generate_structured(prompt)
        return response
    
    def auto_fix_code(self, error):
        prompt = """I have a program with a runtime error.
        Up to last {history_size} CFGs recorded:
        {cfgs}
        Instructions per basic block:
        {instructions}
        Semantic and Type information from source code:
        {sem_ir}

        The program has the following runtime error:
        {error}

        Please provide ONLY actual executable code that fixes this error.
        The code should handle the error case properly following these requirements:
        1. Don't change the overall behavior of the program
        2. Add appropriate safety checks to prevent the error
        3. Return only the code that needs to be injected, no explanations
        
        Format your response as a Python code block starting with ```python and ending with ```
        """
        
        cfg_string = ""
        ins_string = ""
        for module, cfg in self.cfgs.items():
            cfg_history = "None at this time"
            if module in self.__cfg_deque_dict:
                cfg_history = self.__cfg_deque_dict[module].get_cfg_repr()
            cfg_string += f"Module: {module}\n{cfg_history}"
            ins_string += f"Module: {module}\n{cfg.display_instructions()}"
        
        formatted_prompt = prompt.format(
            history_size=self.__cfg_deque_size,
            cfgs=cfg_string,
            instructions=ins_string,
            sem_ir=self.sem_ir.pp(),
            error=error
        )
        
        if self.variable_values is not None:
            formatted_prompt += "\nCurrent variable values at the specified bytecode offset:"
            for module, var_map in self.variable_values.items():
                formatted_prompt += f"\nModule {module}: Offset: {var_map[0]}, Variables: {str(var_map[1])}"
        
        response = self.model.generate(formatted_prompt)
        lines = response.strip().splitlines()
        response = "\n".join(line for line in lines if not line.strip().startswith("```"))
            
        return response
    
    def apply_fix_to_source(self, error, fixed_code):
        try:
            line_number = int(error.get("error_line_number", 0))
            if line_number <= 0:
                print(f"Error: Invalid line number received: {error.get('error_line_number')}")
                return False
        except (ValueError, TypeError):
            print(f"Error: Could not parse line number from error data: {error.get('error_line_number')}")
            return False

        if not fixed_code or not isinstance(fixed_code, str):
             print("Error: Invalid or empty fixed_code received.")
             return False

        print(f"Target line number: {line_number}")
        print(f"Proposed fix code:\n---\n{fixed_code}\n---")

        source_file = self.source_file_path
        if not source_file:
            print("Error: Source file path is not set in ShellGhost instance.")
            return False

        if not os.path.exists(source_file):
            print(f"Error: Source file path does not exist: {source_file}")
            return False

        print(f"Located source file: {source_file}")

        try:
            with open(source_file, 'r', encoding='utf-8') as f: 
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading source file '{source_file}': {e}")
            return False

        if line_number > len(lines):
             print(f"Error: Line number {line_number} is out of bounds for file '{source_file}' (length: {len(lines)} lines).")
             return False

        try:
            original_line_content = lines[line_number - 1]
            indent = original_line_content[:len(original_line_content) - len(original_line_content.lstrip())]
            print(f"Original line ({line_number}): {original_line_content.strip()}")
            print(f"Detected indent: '{indent}' (length: {len(indent)})")

            fixed_lines_indented = []
            for line in fixed_code.strip().splitlines(): 
                fixed_lines_indented.append(indent + line)

            fixed_code_indented = "\n".join(fixed_lines_indented) + "\n"
            print(f"Indented fix code:\n---\n{fixed_code_indented}---")

        except IndexError:
            print(f"Error: Could not access line {line_number} for indentation check (file length {len(lines)}).")
            return False
        except Exception as e:
            print(f"Error during indentation handling: {e}")
            return False


        backup_file = source_file + ".bak"
        try:
            shutil.copy2(source_file, backup_file)
            print(f"Created backup file: {backup_file}")
        except Exception as e:
            print(f"CRITICAL Error: Failed to create backup file '{backup_file}': {e}")
            print("Aborting fix application to prevent data loss.")
            return False 

        lines[line_number - 1] = fixed_code_indented

        try:
            with open(source_file, 'w', encoding='utf-8') as f: 
                f.writelines(lines)
        except Exception as e:
            print(f"Error writing fixed code back to '{source_file}': {e}")
            print(f"Attempting to restore original file from backup '{backup_file}'...")
            try:
                shutil.copy2(backup_file, source_file)
                print("Successfully restored original file from backup.")
            except Exception as restore_e:
                print(f"Failed to write fix AND failed to restore from backup: {restore_e}")
            return False 


        print(f"Successfully applied fix to '{source_file}' at line {line_number}.")
        print("--- Code Fix Application Complete ---")
        return True

    
    def worker(self):
        self.cfg_cv.acquire()
        if self.cfgs is None:
            print("waiting for cfgs")
            self.cfg_cv.wait()
        self.cfg_cv.release()

        current_executing_bbs = {}
        bb_entry_time = {}
        cpu_metrics_history = []
        sample_interval = 1 

        def update_cfg():
            exec_insts = self.tracker.get_exec_inst()

            # don't prompt if there's nothing new
            if not exec_insts:
                return

            for module, offset_list in exec_insts.items():
                try:
                    cfg = self.cfgs[module]

                    if module not in current_executing_bbs:
                        # means start at bb0
                        current_executing_bbs[module] = 0
                        cfg.block_map.idx_to_block[0].exec_count = 1
                        bb_entry_time[module] = time.time()

                    for offset_tuple in offset_list:
                        offset = offset_tuple[0]
                        current_bb = current_executing_bbs[module]
                        if offset not in cfg.block_map.idx_to_block[current_executing_bbs[module]].bytecode_offsets:
                            for next_bb in cfg.edges[current_executing_bbs[module]]:
                                if offset in cfg.block_map.idx_to_block[next_bb].bytecode_offsets:

                                    now = time.time()
                                    if module in bb_entry_time:
                                        elapsed = now - bb_entry_time[module]
                                        cfg.block_map.idx_to_block[current_bb].total_time += elapsed
                                    else:
                                        elapsed = 0
                                    bb_entry_time[module] = now

                                    cfg.edge_counts[(current_executing_bbs[module], next_bb)] += 1
                                    cfg.block_map.idx_to_block[next_bb].exec_count += 1
                                    current_executing_bbs[module] = next_bb
                                    break
                        # sanity check
                        assert offset in cfg.block_map.idx_to_block[current_executing_bbs[module]].bytecode_offsets

                except Exception as e:
                    self.set_finished(e)
                    print(e)
                    break

            self.variable_values = self.tracker.get_variable_values()
            print("Updating cfg deque")
            self.update_cfg_deque(cfg.get_cfg_repr(), module)
            self.logger.info(cfg.to_json())
            print(f"CURRENT INPUTS: {self.tracker.get_inputs()}")

        # -----------
        # NEW FUNCTION: store_memory_usage
        # -----------
        def store_memory_usage():
            print("Updating memory usage")
            """
            Pull memory usage data from the tracker and store or log it.
            You can associate it with your CFG or just log it.
            """
            mem_data = self.tracker.get_memory_usage()
            if not mem_data:
                return  # Nothing to process

            for module, usage_list in mem_data.items():
                # Each usage_list entry is typically (offset, line_no, top_stats)
                # if you're using tracemalloc's snapshot.compare_to().
                print(f"Memory usage data for {module}:")
                for (offset, line_no, top_stats) in usage_list:
                    # Example of logging or storing
                    print(f"[MEM-Usage] Module={module}, offset={offset}, line={line_no}, size_diffs={[top_stat.size_diff for top_stat in top_stats]}")
                    
                    # You can also correlate the offset with a block in self.cfgs[module]
                    # if you want to store memory usage in each block, etc.
        
        def organize_memory_usage_by_bb():
            """
            Given a CFG (with its block_map which includes a set of offsets for each block)
            and memory usage data in the form:
            { module: [ (offset, line_no, top_stats), ... ], ... },
            determine which basic block each memory usage record belongs to,
            and store (or return) a dict mapping block_id to a list of memory usage records.
            """
            usage_by_bb = {}  # dict: block_id -> list of memory snapshots
            mem_data = self.tracker.get_memory_usage()
            # For each module in mem_data (your mem_data keys should match module names used in cfg)
            for module, usage_list in mem_data.items():
                # Get the CFG for the module
                # (Assuming cfg was built per module and accessible as cfg)
                for (offset, line_no, top_stats) in usage_list:
                    # Use a helper: find_block_by_offset() returns the block id (or None) for a given offset.
                    for block_id, block in self.cfgs[module].block_map.idx_to_block.items():
                    # Check if any instruction in this block matches the offset.
                        if any(instr.offset == offset for instr in block.instructions):
                            break
                    if block_id is not None:
                        if block_id not in usage_by_bb:
                            usage_by_bb[block_id] = 0
                        # We convert the top_stats to a serializable (or string) form if needed.
                        # usage_record = {
                        #     "offset": offset,
                        #     "line_no": line_no,
                        #     "top_stats": [str(stat.size_diff) for stat in top_stats]
                        # }
                        # usage_by_bb[block_id].append(usage_record)
                        usage_by_bb[block_id] += sum(stat.size_diff for stat in top_stats)
                    else:
                        print(f"Memory usage at offset {offset} did not map to any basic block.")
            for block_id, usage_records in usage_by_bb.items():
                print(f"Memory usage for block {block_id}:")
                for usage_record in usage_records:
                    print(f"[MEM-Usage] offset={usage_record['offset']}, line={usage_record['line_no']}, size_diffs={usage_record['top_stats']}")
        
        def print_block_timings(cfg):
            """
            Iterate over all blocks in the CFG and print their execution count and total runtime.
            """
            for block_id, block in cfg.block_map.idx_to_block.items():
                print(f"Block bb{block_id}: exec_count = {block.exec_count}, total_time = {block.total_time:.4f} seconds")

        def sample_cpu_io_metrics() -> dict:
            """
            Sample of some cpu utilization metrics.
            """
            total_cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_times = psutil.cpu_times_percent(interval=0.1)
            cpu_freq = psutil.cpu_freq()
            cpu_freq_dict = cpu_freq._asdict() if cpu_freq is not None else None

            io_counters = psutil.disk_io_counters()
            read_bytes = io_counters.read_bytes 
            write_bytes = io_counters.write_bytes

            return {
                'total_cpu_percent': total_cpu_percent,
                'cpu_times': cpu_times._asdict(),
                'cpu_freq': cpu_freq_dict,
                'io_read_bytes' : read_bytes,
                'io_write_bytes' : write_bytes,
                'read_count': io_counters.read_count,
                'write_count': io_counters.write_count,
                }

        with self.finished_exception_lock:
            while not self.finished:
                self.finished_exception_lock.release()

                time.sleep(3)
                print("\nUpdating cfgs")
                update_cfg()
                # store_memory_usage()  # <-- Call the new function here
                organize_memory_usage_by_bb()
                
                # cpu and io metrics that might be helpful to the llm
                cpu_sample = sample_cpu_io_metrics()
                cpu_metrics_history.append(cpu_sample)
                print("CPU Metrics Sample:", cpu_sample)

                self.finished_exception_lock.acquire()

        for module, current_bb in current_executing_bbs.items():
            now = time.time()
            if module in bb_entry_time:
                elapsed = now - bb_entry_time[module]
                self.cfgs[module].block_map.idx_to_block[current_bb].total_time += elapsed


        print("\nUpdating cfgs at the end")
        update_cfg()
        # store_memory_usage()  # one last call at the end
        organize_memory_usage_by_bb()

        response = self.prompt_for_runtime()
        #going to move this into fardeen's implementation
        print("\nBlock timing information:")
        for module, cfg in self.cfgs.items():
            print(f"Module: {module}")
            print_block_timings(cfg)

        print(response["behavior_description"])
        print(response["error_list"])
        repo_path = os.path.abspath(os.path.expanduser("~/UMich/jaseci-ginS"))
        ghostwriter = GhostWriter(
            self.model,
            repo_path=repo_path,
            github_token=os.getenv("GITHUB_TOKEN"),
            github_repo_fullname="wilkincr/jaseci-ginS",
        )
        ghostwriter.rewrite_content(response["error_list"], "jac/examples/gins_scripts/example.jac")

