"""Genrate a control flow graph from genarated bytecode.

This pass generates a control flow graph from the bytecode generated by the previous pass.

Code was initially inspired by:
https://bernsteinbear.com/blog/discovering-basic-blocks/

The rest was extended to handle some nuance and edge conditions that were not considered.
"""
import marshal
import dis
from collections import defaultdict
from typing import List, Optional, Iterator
#import graphviz
#from graphviz import Digraph
CODEUNIT_SIZE = 2
class BytecodeOp:
    def __init__(self, op: int, arg: int, offset: int, argval:int, is_jump_target: bool) -> None:
        self.op = op
        self.arg = arg
        self.offset = offset
        self.argval = argval
        self.is_jump_target= is_jump_target

    def __repr__(self):
        return f"{self.offset}: f{self.op} - {self.arg} - {self.argval}"
    def is_branch(self) -> bool:
        return self.op in {
            "JUMP_ABSOLUTE",
            "JUMP_FORWARD",
            "POP_JUMP_IF_TRUE",
            "POP_JUMP_IF_FALSE",
            "JUMP_IF_TRUE_OR_POP",
            "JUMP_IF_FALSE_OR_POP",
        }
    def is_relative_branch(self) -> bool:
        return self.op in {
            "FOR_ITER",
            "JUMP_FORWARD",
        }    
    def is_return(self) -> bool:
        return self.op == "RETURN_VALUE"

    def is_raise(self) -> bool:
        return self.op == "RAISE_VARARGS"

class Block:
    def __init__(self, id: int, instructions: List):
        self.id: int = id
        self.instructions = instructions
    def __repr__(self):
      instructions = "\n".join([str(instr) for instr in self.instructions])
      return f"bb{self.id}:\n{instructions}"

class BlockMap:
    def __init__(self) -> None:
        self.idx_to_block: Dict[int, Block] = {}

    def add_block(self, idx, block):
        self.idx_to_block[idx] = block

    def __repr__(self) -> str:
        result = []
        for block in self.idx_to_block.values():
          result.append(repr(block))
        return "\n".join(result)
    def __str__(self) -> str:
        return self.__repr__()

def disassemble_bytecode(bytecode):
    code_object = marshal.loads(bytecode)
    instructions = []
    for i, instr in enumerate(dis.get_instructions(code_object)):
      instructions.append(BytecodeOp(
        op = instr.opname, 
        arg=instr.arg,
        offset=instr.offset,
        argval=instr.argval,
        is_jump_target=instr.is_jump_target
        ))
    return instructions

def create_BBs(instructions: List[BytecodeOp]) -> BlockMap:
    block_starts = set([0])
    block_map = BlockMap()
    num_instr = len(instructions)
    
    # Create offset to index mapping
    offset_to_index = {instr.offset: idx for idx, instr in enumerate(instructions)}
    max_offset = instructions[-1].offset + CODEUNIT_SIZE

    def valid_offset(offset):
        return offset >= 0 and offset <= max_offset
    # Identify all block starts
    for instr in instructions:
        if instr.is_branch():
            next_instr_offset = instr.offset + CODEUNIT_SIZE
            if valid_offset(next_instr_offset):
                block_starts.add(next_instr_offset)
            
            if instr.is_relative_branch():
                target_offset = instr.offset + instr.argval
            else:
                target_offset = instr.argval
                
            if valid_offset(target_offset):
                block_starts.add(target_offset)
        
        if instr.is_jump_target:
            block_starts.add(instr.offset)

    block_starts_ordered = sorted(block_starts)

    
    for block_id, start_offset in enumerate(block_starts_ordered):
        start_index = offset_to_index[start_offset]
        end_index = num_instr
        
        # Find the corresponding end_index
        for offset in block_starts_ordered:
            if offset > start_offset:
                end_index = offset_to_index[offset]
                break
        
        # Collect instructions for this block
        block_instrs = instructions[start_index:end_index]
        block_map.add_block(block_id, Block(block_id, block_instrs))
    
    return block_map


class CFG:
    def __init__(self):
        self.nodes = set()
        self.edges = {}

    def add_node(self, node_id):
        self.nodes.add(node_id)
        if node_id not in self.edges:
            self.edges[node_id] = []

    def add_edge(self, from_node, to_node):
        if from_node in self.edges:
            self.edges[from_node].append(to_node)
        else:
            self.edges[from_node] = [to_node]

    def __repr__(self):
        result = []
        for node in self.nodes:
            result.append(f'Node bb{node}:')
            if node in self.edges and self.edges[node]:
                for succ in self.edges[node]:
                    result.append(f'  -> bb{succ}')
        return "\n".join(result)
def create_cfg(block_map: BlockMap) -> CFG:
    cfg = CFG()

    for block_id, block in block_map.idx_to_block.items():
        cfg.add_node(block_id)
        
        last_instr = block.instructions[-1]

        # Handle conditional jumps (e.g., POP_JUMP_IF_FALSE)
        if last_instr.is_branch():
            target_offset = last_instr.argval if not last_instr.is_relative_branch() else (last_instr.offset + last_instr.argval)
            target_block = find_block_by_offset(block_map, target_offset)
            if target_block is not None:
                cfg.add_edge(block_id, target_block)
            # Fall-through to next block if it's a conditional branch
            if last_instr.op.startswith('POP_JUMP_IF'):
                fall_through_offset = block.instructions[-1].offset + CODEUNIT_SIZE
                fall_through_block = find_block_by_offset(block_map, fall_through_offset)
                if fall_through_block is not None:
                    cfg.add_edge(block_id, fall_through_block)

        # Handle unconditional jumps (e.g., JUMP_FORWARD, JUMP_ABSOLUTE)
        elif last_instr.op.startswith("JUMP"):
            target_offset = last_instr.argval if not last_instr.is_relative_branch() else (last_instr.offset + last_instr.argval)
            target_block = find_block_by_offset(block_map, target_offset)
            if target_block is not None:
                cfg.add_edge(block_id, target_block)

        # Handle fall-through to the next block for non-control flow instructions
        else:
            fall_through_offset = block.instructions[-1].offset + CODEUNIT_SIZE
            fall_through_block = find_block_by_offset(block_map, fall_through_offset)
            if fall_through_block is not None:
                cfg.add_edge(block_id, fall_through_block)

    return cfg

def find_block_by_offset(block_map: BlockMap, offset: int) -> int:
    for block_id, block in block_map.idx_to_block.items():
        if any(instr.offset == offset for instr in block.instructions):
            return block_id
    return None

# Function to visualize CFG using Graphviz
def visualize_cfg(cfg: CFG):
    dot = Digraph(comment="Control Flow Graph")
    for node in cfg.nodes:
        dot.node(f"bb{node}", f"BB{node}")
    for from_node, to_nodes in cfg.edges.items():
        for to_node in to_nodes:
            dot.edge(f"bb{from_node}", f"bb{to_node}")
    return dot

# Sample list of instructions for processing
##simple=
instructions = disassemble_bytecode(b'c\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x01\xf3*\x00\x00\x00\x97\x00d\x00d\x01l\x00m\x01Z\x01\x01\x00d\x00Z\x02d\x00Z\x03e\x03d\x00k\\\x00\x00r\x02d\x02Z\x02d\x03Z\x02y\x04)\x05\xe9\x00\x00\x00\x00)\x01\xda\x0bannotations\xe9\x01\x00\x00\x00\xe9\xff\xff\xff\xffN)\x04\xda\n__future__r\x02\x00\x00\x00\xda\x01a\xda\x01x\xa9\x00\xf3\x00\x00\x00\x00\xfaP/Users/jakobtherkelsen/Documents/jaseci-ginS/jac/examples/ginsScripts/simple.jac\xda\x08<module>r\x0b\x00\x00\x00\x01\x00\x00\x00s%\x00\x00\x00\xf0\x03\x01\x01\x01\xf5\x02\x07\x02\x03\xd8\x05\x06\x801\xd8\x05\x06\x801\xd8\x06\x07\x881\x82f\xd8\x07\x08\x80Q\xe0\x05\x07\x811r\t\x00\x00\x00')
#hot path
#instructions = disassemble_bytecode(b'c\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x01\xf3T\x00\x00\x00\x97\x00d\x00d\x01l\x00m\x01Z\x01\x01\x00d\x00Z\x02d\x00Z\x03e\x02d\x02k\x02\x00\x00r\x19e\x02d\x03z\x06\x00\x00d\x00k(\x00\x00r\x03d\x04Z\x03n\x02d\x03Z\x03e\x02d\x04z\r\x00\x00Z\x02e\x02d\x02k\x02\x00\x00r\x01\x8c\x18y\x05y\x05)\x06\xe9\x00\x00\x00\x00)\x01\xda\x0bannotations\xe9\x0f\x00\x00\x00\xe9\x02\x00\x00\x00\xe9\x01\x00\x00\x00N)\x04\xda\n__future__r\x02\x00\x00\x00\xda\x01a\xda\x01b\xa9\x00\xf3\x00\x00\x00\x00\xfaR/Users/jakobtherkelsen/Documents/jaseci-ginS/jac/examples/ginsScripts/hot_path.jac\xfa\x08<module>r\x0c\x00\x00\x00\x01\x00\x00\x00sD\x00\x00\x00\xf0\x03\x01\x01\x01\xf5\x02\x0c\x02\x03\xd8\x07\x08\x801\xd8\x07\x08\x801\xd8\t\n\x88R\x8a\x16\xd8\x08\t\x88A\x89\x05\x90\x11\x8a\n\xd8\x0b\x0c\x81q\xf0\x06\x00\x0c\r\x80q\xe0\x05\x06\x88!\x81W\x80Q\xf0\x0f\x00\n\x0b\x88R\x8d\x16r\n\x00\x00\x00')
BBs = create_BBs(instructions)
print(BBs)

cfg = create_cfg(BBs)
print("\nControl Flow Graph (CFG):")
print(cfg)

# Visualize CFG
# dot = visualize_cfg(cfg)
# dot.render('cfg.gv', view=True)