from __future__ import annotations

import dis
from dataclasses import dataclass
from typing import List, Optional, Set, Any, Dict, Tuple
from typing_extensions import Protocol

import networkx as nx  # XXX: remove need for NX later

from .controlflow import CFGraph

try:
    import graphviz as gv
except ImportError:
    pass


def translate(code):
    bc = dis.Bytecode(code)
    blocks = translate_bytecode_to_blocks(bc)
    for blk in blocks:
        print('\n'.join(blk.dump_lines()))


    loop_restructuring(blocks)
    gv_render_blocks(blocks).view()


def loop_restructuring(blocks: List[Block]):
    # RVSDG Loop restructuring.
    loops = compute_unstructured_loop(blocks)
    _loop_restructuring_details(blocks, loops)

def compute_unstructured_loop(blocks):
    scc_list = compute_scc_blocks(blocks)
    loops = [scc for scc in scc_list if len(scc) > 1]
    return loops

def _loop_restructuring_details(blocks: List[Block], loops: List[Set[str]]):
    cfg = _build_cfg(blocks)
    cfg.render_dot(filename='cfg.dot').view()

    blockmap = {b.key: b for b in blocks}

    for loop in loops:
        print("====loop", loop)
        entry_arcs: Set[Tuple[str, str]] = {
            (pred, node) for node in loop
            for pred, _data in cfg.predecessors(node)
            if pred not in loop
        }
        entry_vertices = {d for _s, d in entry_arcs}

        exit_arcs: Set[Tuple[str, str]] = {
            (node, succ) for node in loop
            for succ, _data in cfg.successors(node)
            if succ not in loop
        }
        exit_vertices = {d for _s, d in exit_arcs}

        repetition_arcs: Set[Tuple[str, str]] = {
            (node, succ) for node in loop
            for succ, _data in cfg.successors(node)
            if succ in entry_vertices
        }

        print("entry_arcs", entry_arcs)
        print("entry_vertices", entry_vertices)
        print("exit_arcs", exit_arcs)
        print("exit_vertices", exit_vertices)
        print("repetition_arcs", repetition_arcs)
        [loop_entry] = entry_vertices

        # FIXME: there should be a single control-point
        assert len(entry_arcs) == 1, len(entry_arcs)
        if len(exit_arcs) > 1:

            merged_exit_outside_blk = Block(f"{loop_entry}.join.exit.outside")
            merged_exit_outside_blk.body.append(OpSwitch())
            blocks.append(merged_exit_outside_blk)

            merged_exit_blk = Block(f"{loop_entry}.join.exit")
            merged_exit_blk.body.append(OpJump(merged_exit_outside_blk.key))
            blocks.append(merged_exit_blk)

            for ex_src, ex_dst in exit_arcs:
                tmp_blk = Block(f"{ex_src}.exit.state")
                tmp_blk.body.append(OpControlPoint("LOOP_STATE", ex_dst))
                tmp_blk.body.append(OpJump(merged_exit_blk.key))
                blocks.append(tmp_blk)

                merged_exit_outside_blk.terminator.targets.append(ex_dst)

                ex_src_blk = blockmap[ex_src]
                ex_src_blk.terminator.replace_target(ex_dst, tmp_blk.key)

        loop_rep_blk = Block(f"{loop_entry}.join.backedge")
        loop_rep_blk.body.append(OpLoopRepeat(loop_entry))
        blocks.append(loop_rep_blk)
        for ra_src, ra_dst in repetition_arcs:
            assert loop_entry == ra_dst
            src_block : Block = blockmap[ra_src]
            src_block.terminator.replace_target(ra_dst, loop_rep_blk.key)

        # recursively restructure nested loop
        subgraph = set(loop) - entry_vertices - exit_vertices
        loops = compute_unstructured_loop([blockmap[k] for k in subgraph])
        print("inner loops", loops)
        # FIXME: this doesn't need to recurse. can modify loops perhaps
        _loop_restructuring_details(blocks, loops)


def _build_cfg(blocks: List[Block]) -> CFGraph:
    cfg = CFGraph()
    for node in blocks:
        cfg.add_node(node.key)
    for node in blocks:
        src = node.key
        for dst in node.get_outgoings():
            cfg.add_edge(src, dst)
    cfg.set_entry_point(str(0))
    cfg.process()
    return cfg

def compute_scc_blocks(blocks: List[Block]) -> List[Set[str]]:
    g = nx.DiGraph()
    for blk in blocks:
        g.add_node(blk.key)
    for blk in blocks:
        for edge in blk.get_outgoings():
            g.add_edge(blk.key, edge)

    scc = list(nx.strongly_connected_components(g))
    return scc


def translate_bytecode_to_blocks(bc: dis.Bytecode) -> List[Block]:
    print(bc.dis())

    manager = BlockManager()

    next_offset = _build_next_offset(bc)

    lastblk = None
    for bcinst in bc:
        if bcinst.is_jump_target:
            blk = manager.get_block(bcinst.offset)
            manager.last = blk
        else:
            blk = manager.at(bcinst.offset)
            if blk.is_terminated():
                blk = manager.get_block(bcinst.offset)
                manager.last = blk
        body = blk.body
        # Terminate last block
        if lastblk and blk is not lastblk and not lastblk.is_terminated():
            lastblk.body.append(OpJump(blk.key))
        lastblk = blk
        # Process instructions
        if bcinst.opname == "FOR_ITER":
            end = manager.get_block(bcinst.argval)
            loop = manager.get_block(next_offset[bcinst.offset])
            body.append(OpFor(loop.key, end.key))

            manager.last = loop
        elif "JUMP_IF_" in bcinst.opname:
            bbthen = manager.get_block(bcinst.argval)
            bbelse = manager.get_block(next_offset[bcinst.offset])
            manager.last = bbelse
            bb1, bb0 = (bbthen, bbelse) if "IF_TRUE" in bcinst.opname else (bbelse, bbthen)
            body.append(OpJumpIf(bb1.key, bb0.key))
        elif bcinst.opname in {"JUMP_FORWARD", "JUMP_ABSOLUTE"}:
            body.append(OpJump(manager.get_block(bcinst.argval).key))
        elif bcinst.opname == "RETURN_VALUE":
            body.append(OpRet())
        else:
            body.append(OpBcInst(bcinst))

    # prune dead blocks
    reachable = {str(0)}
    for blk in manager.blocks.values():
        reachable |= set(blk.get_outgoings())

    blocks = [blk for blk in manager.blocks.values() if blk.key in reachable]
    verify_blocks(blocks)
    return blocks


def verify_blocks(blocks: List[Block]):
    for blk in blocks:
        blk.verify()


def _build_next_offset(bc) -> Dict[int, int]:
    seq = list(bc)
    nxt = seq[1:]
    out = {bcinst.offset: bcnext.offset for bcinst, bcnext in zip(seq, nxt)}
    return out


class BlockManager:
    def __init__(self):
        cur = Block(str(0))
        self.last = cur
        self.blocks : Dict[str, Block] = {cur.key: cur}
        self.jmptable: Dict[int, str] = {0: cur.key}

    def new_block(self, offset: int) -> Block:
        block = Block(str(offset))
        assert block.key not in self.blocks, block.key
        self.blocks[block.key] = block
        self.jmptable[offset] = block.key
        return block

    def get_block(self, offset: int) -> Block:
        if offset in self.jmptable:
            return self.blocks[self.jmptable[offset]]
        else:
            blk = self.new_block(offset)
            self.jmptable[offset] = blk.key
            return blk

    def at(self, offset: int) -> Block:
        k = self.jmptable.get(offset, self.last.key)
        out = self.blocks[k]
        self.last = out
        return out


class Block:
    def __init__(self, key: str):
        self.key = key
        self.body : List[Operation] = []

    def dump_lines(self) -> List[str]:
        buf: List[str] = [f"Block({self.key}) {{"]
        for inst in self.body:
            buf.extend(_indent_lines(inst.dump_lines()))
        buf.append("}")
        return buf

    def is_terminated(self) -> bool:
        return bool(self.body) and isinstance(self.body[-1], OpTerminator)

    @property
    def terminator(self) -> OpTerminator:
        if not self.body:
            raise ValueError(f"{self} is empty")
        last = self.body[-1]
        if isinstance(last, OpTerminator):
            return last
        else:
            raise ValueError("block not terminated")

    def get_outgoings(self) -> List[str]:
        return self.terminator.get_targets()

    def verify(self) -> None:
        # check for single terminator in body
        assert self.is_terminated()
        for inst in self.body[:-1]:
            assert not isinstance(inst, OpTerminator)


def gv_render_blocks(blocks: List[Block]):
    g = gv.Digraph()
    for blk in blocks:
        label = '\l'.join(blk.dump_lines()) + '\l'
        g.node(blk.key, shape='rect', label=label)

    for blk in blocks:
        for og in blk.get_outgoings():
            g.edge(blk.key, og)
    return g

def _indent_lines(lines: List[str], prefix=' ' * 4) -> List[str]:
    return [f"{prefix}{ln}" for ln in lines]


class Operation(Protocol):
    def dump_lines(self) -> List[str]: ...



class OpBcInst(Operation):
    def __init__(self, bcinst: dis.Instruction):
        self.bcinst = bcinst

    def dump_lines(self) -> List[str]:
        bcinst = self.bcinst
        clsname = self.__class__.__name__
        return [f"{clsname}:{bcinst.opname}({ bcinst.argval})[@{bcinst.offset}]"]


# class OpBcMicro(Operation):
#     def __init__(self, opname: str, argval: Any):
#         self.opname = opname
#         self.argval = argval

#     def dump_lines(self) -> List[str]:
#         clsname = self.__class__.__name__
#         return [f"{clsname}:{self.opname}({self.argval})"]



class OpControlPoint(Operation):
    def __init__(self, opname: str, argval: Any):
        self.opname = opname
        self.argval = argval

    def dump_lines(self) -> List[str]:
        clsname = self.__class__.__name__
        return [f"{clsname}:{self.opname}({self.argval})"]


class TerminatorTrait(Protocol):
    def get_targets(self) -> List[str]: ...

    def replace_target(self, old: str, new: str): ...

class OpTerminator(TerminatorTrait, Operation):
    pass

class OpJump(OpTerminator):
    def __init__(self, target: str):
        self.target = target

    def dump_lines(self) -> List[str]:
        clsname = self.__class__.__name__
        return [f"{clsname}({self.target})"]

    def get_targets(self) -> List[str]:
        return [self.target]

    def replace_target(self, old: str, new: str):
        if self.target == old:
            self.target = new

class OpJumpIf(OpTerminator):
    def __init__(self, then_target: str, else_target: str):
        self.then_target = then_target
        self.else_target = else_target

    def dump_lines(self) -> List[str]:
        clsname = self.__class__.__name__
        return [f"{clsname}({self.then_target}, {self.else_target})"]

    def get_targets(self) -> List[str]:
        return [self.then_target, self.else_target]

    def replace_target(self, old: str, new: str):
        if self.then_target == old:
            self.then_target = new
        if self.else_target == old:
            self.else_target = new

class OpRet(OpTerminator):
    def dump_lines(self) -> List[str]:
        clsname = self.__class__.__name__
        return [f"{clsname}"]

    def get_targets(self) -> List[str]:
        return []

    def replace_target(self, old: str, new: str):
        pass   # do nothing


class OpFor(OpTerminator):
    def __init__(self, loop_target: str, end_target: str):
        self.loop_target = loop_target
        self.end_target = end_target

    def dump_lines(self) -> List[str]:
        clsname = self.__class__.__name__
        return [f"{clsname}({self.loop_target}, {self.end_target})"]

    def get_targets(self) -> List[str]:
        return [self.loop_target, self.end_target]

    def replace_target(self, old: str, new: str):
        if self.loop_target == old:
            self.loop_target = new
        if self.end_target == old:
            self.end_target = new


class OpLoopRepeat(OpTerminator):

    def __init__(self, loop_head: str):
        self.loop_head = loop_head

    def dump_lines(self) -> List[str]:
        clsname = self.__class__.__name__
        return [f"{clsname}({self.loop_head})"]

    def get_targets(self) -> List[str]:
        return []



class OpSwitch(OpTerminator):

    def __init__(self):
        self.targets : List[str] = []

    def dump_lines(self) -> List[str]:
        clsname = self.__class__.__name__
        return [f"{clsname} {self.targets}"]

    def get_targets(self) -> List[str]:
        return list(self.targets)
