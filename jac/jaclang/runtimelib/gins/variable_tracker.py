from collections import defaultdict

class VariableTracker:
    def __init__(self, curr_variable_values, line_to_blocks, top_k=5):
        self.variables = defaultdict(                                   # module
                lambda: defaultdict(                         # basic‑block id
                    lambda: defaultdict(                     # variable
                        lambda: defaultdict(int)             # value → count
                    )   
                )
            )
        
        for mod, lines in curr_variable_values.items():             # module
            for line_no, vars_at_line in lines.items():             # source line
                # grab **all** blocks that contain that line
                bb_id = line_to_blocks[line_no]
                for var, value_counts in vars_at_line.items():
                    for value, cnt in value_counts.items():
                        self.variables[mod][bb_id][var][value] += cnt

        self.top_k = top_k
    
    def __repr__(self):
        result = []
        for mod, bb_dict in self.variables.items():
            result.append(f"module: {mod}")
            for bb_id, var_dict in bb_dict.items():
                result.append(f"bb{bb_id}")
                for var, val_counts in var_dict.items():
                    top_vals = sorted(val_counts.items(),
                                    key=lambda x: -x[1])[:self.top_k]
                    for val, freq in top_vals:
                        result.append(f"{var} = {val}  (seen {freq}×)")

        return "\n".join(result)
        