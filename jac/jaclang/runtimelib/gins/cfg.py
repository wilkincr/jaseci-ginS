import marshal
import dis
from collections import defaultdict
from typing import List, Optional, Iterator

class BytecodeOp:
    def __init__(
        self,
        op: int,
        arg: int,
        offset: int,
        argval: int,
        argrepr: str,
        is_jump_target: bool,
        lineno: Optional[int] = None
    ) -> None:
        self.op = op
        self.arg = arg
        self.offset = offset
        self.argval = argval
        self.argrepr = argrepr
        self.is_jump_target = is_jump_target
        self.lineno = lineno
        self.__offset_size = 0

    def __repr__(self):
        return (
            f"Instr: offset={self.offset}, Lineno={self.lineno}, "
            f"Opname={self.op}, arg={self.arg}, argval={self.argval}, "
            f"argrepr={self.argrepr}, jump_t={self.is_jump_target}"
        )

    def is_branch(self) -> bool:
        return self.op in {
            "JUMP_ABSOLUTE",
            "JUMP_FORWARD",
            "JUMP_BACKWARD",
            "POP_JUMP_IF_TRUE",
            "POP_JUMP_IF_FALSE",
            "JUMP_IF_TRUE_OR_POP",
            "JUMP_IF_FALSE_OR_POP",
        }

    def is_conditional_branch(self) -> bool:
        return self.op in {
            "JUMP_IF_TRUE_OR_POP",
            "JUMP_IF_FALSE_OR_POP",
            "POP_JUMP_IF_TRUE",
            "POP_JUMP_IF_FALSE",
        }

    def is_relative_branch(self) -> bool:
        return False

    def is_return(self) -> bool:
        return self.op == "RETURN_VALUE"

    def is_raise(self) -> bool:
        return self.op == "RAISE_VARARGS"

    def is_for_iter(self) -> bool:
        return self.op == "FOR_ITER"

    def set_offset_size(self, size) -> None:
        self.__offset_size = size

    def get_offset_size(self) -> int:
        return self.__offset_size

    def get_next_instruction_offset(self) -> int:
        return self.__offset_size + self.offset


class Block:
    def __init__(self, id: int, instructions: List[BytecodeOp]):
        self.id: int = id
        self.instructions = instructions
        self.exec_count = 0
        self.bytecode_offsets = set(
            [instr.offset for instr in self.instructions if instr.offset is not None]
        )

    def __repr__(self):
        instructions = "\n".join([str(instr) for instr in self.instructions])
        return f"bb{self.id}:\n{instructions}"


class BlockMap:
    def __init__(self) -> None:
        self.idx_to_block: dict[int, Block] = {}

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
        line = getattr(instr, "lineno", None)
        if line is None:
            line = instr.starts_line
        b_op = BytecodeOp(
            op=instr.opname,
            arg=instr.arg,
            offset=instr.offset,
            argval=instr.argval,
            argrepr=instr.argrepr,
            is_jump_target=instr.is_jump_target,
            lineno=line,
        )
        instructions.append(b_op)

        # set offset_size for calculating next instruction offset
        if i != 0:
            prev_instruction = instructions[i - 1]
            prev_instruction.set_offset_size(instr.offset - prev_instruction.offset)
    return instructions


def create_BBs(instructions: List[BytecodeOp]) -> BlockMap:
    block_starts = set([0])
    block_map = BlockMap()
    offset_to_index = {instr.offset: idx for idx, instr in enumerate(instructions)}
    max_offset = instructions[-1].get_next_instruction_offset()

    def valid_offset(offset):
        return 0 <= offset <= max_offset

    # Identify block starts
    for instr in instructions:
        if instr.is_branch() or instr.is_for_iter():
            next_instr_offset = instr.get_next_instruction_offset()
            target_offset = instr.argval

            if instr.is_for_iter():
                block_starts.add(instr.offset)
            elif valid_offset(next_instr_offset):
                block_starts.add(next_instr_offset)

            if valid_offset(target_offset):
                block_starts.add(target_offset)

    # Build each block from start_offset -> next start offset
    block_starts_ordered = sorted(block_starts)
    for block_id, start_offset in enumerate(block_starts_ordered):
        end_offset = (
            block_starts_ordered[block_id + 1]
            if block_id + 1 < len(block_starts_ordered)
            else instructions[-1].get_next_instruction_offset()
        )
        start_index = offset_to_index[start_offset]
        end_index = offset_to_index.get(end_offset, len(instructions) - 1)

        if start_index == end_index:
            end_index += 1

        block_instrs = instructions[start_index:end_index]
        block_map.add_block(block_id, Block(block_id, block_instrs))
    return block_map


class CFG:
    def __init__(self, block_map: BlockMap):
        self.nodes = set()
        self.edges = {}
        self.edge_counts = {}
        self.block_map = block_map

        # Optional: store memory usage by block
        # dict of block_id -> list of memory usage stats
        self.memory_usage = defaultdict(list)

    def add_node(self, node_id):
        self.nodes.add(node_id)
        if node_id not in self.edges:
            self.edges[node_id] = []

    def add_edge(self, from_node, to_node):
        if from_node in self.edges:
            self.edges[from_node].append(to_node)
        else:
            self.edges[from_node] = [to_node]
        self.edge_counts[(from_node, to_node)] = 0

    def display_instructions(self):
        return repr(self.block_map)

    def get_cfg_repr(self):
        return self.__repr__()

    def to_json(self):
        obj = {"cfg_bbs": []}
        for node in self.nodes:
            bb_obj = {
                "bb_id": node,
                "freq": self.block_map.idx_to_block[node].exec_count,
                "predicted_edges": [],
                "actual_edges": [],
            }
            if node in self.edges and self.edges[node]:
                for succ in self.edges[node]:
                    edge_placeholder = {"edge_to_bb_id": succ, "freq": 0}
                    bb_obj["predicted_edges"].append(edge_placeholder)
                    edge_obj = {
                        "edge_to_bb_id": succ,
                        "freq": self.edge_counts[(node, succ)],
                    }
                    bb_obj["actual_edges"].append(edge_obj)
            obj["cfg_bbs"].append(bb_obj)
        return obj

    def record_memory_usage(self, block_id: int, usage_info: dict):
        """Optionally store memory usage info for the given block."""
        self.memory_usage[block_id].append(usage_info)

    def __repr__(self):
        result = []
        for node in self.nodes:
            freq = self.block_map.idx_to_block[node].exec_count
            result.append(f"Node bb{node} (freq={freq}):")
            if node in self.edges and self.edges[node]:
                for succ in self.edges[node]:
                    edge_freq = self.edge_counts[(node, succ)]
                    result.append(f"(freq={edge_freq})-> bb{succ}")
        return "\n".join(result)


def create_cfg(block_map: BlockMap) -> CFG:
    cfg = CFG(block_map)
    for block_id, block in block_map.idx_to_block.items():
        cfg.add_node(block_id)
        first_instr = block.instructions[0]
        last_instr = block.instructions[-1]

        if first_instr.is_for_iter():
            target_offset = first_instr.argval
            end_for_block = find_block_by_offset(block_map, target_offset)
            if end_for_block is not None:
                cfg.add_edge(block_id, end_for_block)

        # handle jumps
        if last_instr.is_branch():
            target_offset = last_instr.argval
            target_block = find_block_by_offset(block_map, target_offset)
            if target_block is not None:
                cfg.add_edge(block_id, target_block)
            if last_instr.is_conditional_branch():
                fall_through_offset = last_instr.get_next_instruction_offset()
                fall_through_block = find_block_by_offset(block_map, fall_through_offset)
                if fall_through_block is not None:
                    cfg.add_edge(block_id, fall_through_block)
        else:
            # handle fall through
            fall_through_offset = last_instr.get_next_instruction_offset()
            fall_through_block = find_block_by_offset(block_map, fall_through_offset)
            if (
                fall_through_block is not None
                and fall_through_offset != last_instr.offset
            ):
                cfg.add_edge(block_id, fall_through_block)
    return cfg


def find_block_by_offset(block_map: BlockMap, offset: int) -> Optional[int]:
    for block_id, block in block_map.idx_to_block.items():
        if any(instr.offset == offset for instr in block.instructions):
            return block_id
    return None


def visualize_cfg(cfg: CFG):
    try:
        from graphviz import Digraph

        dot = Digraph(comment="Control Flow Graph")
        for node in cfg.nodes:
            dot.node(f"bb{node}", f"BB{node}")
        for from_node, to_nodes in cfg.edges.items():
            for to_node in to_nodes:
                dot.edge(f"bb{from_node}", f"bb{to_node}")
        return dot
    except ImportError:
        print("Graphviz not installed, can't visualize CFG")
        return None
