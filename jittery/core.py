from __future__ import annotations
from collections import defaultdict, ChainMap
import dis
from os import name
import weakref
from dataclasses import dataclass
from prettyprinter import pprint, install_extras
from typing import List, Optional, Set

try:
    import graphviz as gv
except ImportError:
    pass

# install_extras(exclude=["django", "ipython"])


def translate(code):
    bc = dis.Bytecode(code)
    print(bc.dis())

    processed = {}
    traced = trace_control(list(bc), processed)
    print('-' * 80)
    traced.dump()
    traced.show_graphviz()

    # build graph
    return _process_cfg(traced)

def _build_cfg(bcregion):
    from .controlflow import CFGraph

    nodes : Set[BCRegion] = set()
    edges = set()
    dfs_stack : List[BCRegion] = [bcregion]
    while dfs_stack:
        tos = dfs_stack.pop()
        nodes.add(tos)
        for other in tos.edges:
            edges.add((tos, other))
            if other not in nodes:
                dfs_stack.append(other)

    cfg = CFGraph()
    for node in nodes:
        cfg.add_node(name_node(node))
    for x, y in edges:
        cfg.add_edge(name_node(x), name_node(y))
    cfg.set_entry_point(name_node(bcregion))
    cfg.process()
    return cfg

def _process_cfg(bcregion):
    # group while-loops
    loops = {}
    cfg = _build_cfg(bcregion)
    cfg_loops = cfg.loops()
    for loop in cfg_loops.values():
        header = loop.header
        body = loop.body
        loops[header] = body
    _render_loops(cfg, loops)

    # replace loops
    processed = {}
    out = _replace_loops(bcregion, loops, processed)
    out = _expand_branches(out)
    out.show_graphviz()
    assert False, """
    Not sure how to expand and matches if-elses
    """
    # return out

def _expand_branches(bcregion: BCRegion) ->  BCRegion:
    cfg = _build_cfg(bcregion)
    pdoms = cfg.post_dominators()

    common = set()
    for node in cfg.nodes():
        if len(node.edges) > 1:
            pdom_node = pdoms[node]
            endif_set = pdom_node - {node}
            if endif_set:
                endif = min(endif_set, key=lambda x: x.body[0].offset)
                common.add(endif)

    print('common', common)
    def expand(root):
        if root in common:
            return root
        else:
            new_edges = list(map(expand, root.edges))
            return BCRegion(body=root.body, edges=new_edges)

    out =  expand(bcregion)
    return out



def _replace_loops(bcregion, loops, processed) -> BCRegion:
    if bcregion in processed:
        return processed[bcregion]
    key = name_node(bcregion)
    if key not in loops:
        out = BCRegion(
            body=bcregion.body,
            edges=[_replace_loops(e, loops, processed) for e in bcregion.edges]
        )
    else:
        body = loops[key]
        [endloop] = set(map(name_node, bcregion.edges)) - set(body)
        [end] = [x for x in bcregion.edges if name_node(x) == endloop]
        first = bcregion.body[0]
        repl = dis.Instruction(
            opname="MY_LOOP", opcode=None, arg=None, argval=None, argrepr="",
            offset=first.offset, starts_line=first.starts_line,
            is_jump_target=first.is_jump_target)
        out = BCRegion(body=[repl], edges=[end])
    processed[bcregion] = out
    return out


def name_node(bcregion: BCRegion):
    return bcregion
    # first = bcregion.body[0]
    # term = bcregion.terminator
    # out = f"Node[{first.offset:06} {first.opname} ... {term.opname}]"
    # return out

def _render_loops(cfg, loops):
    #########
    # render graph

    # Start from the outermost loops.
    todos = [(k, loops[k]) for k in cfg.topo_order() if k in loops]
    drawn = set()

    def draw(g, todos):
        root, children = todos.pop(0)
        with g.subgraph(name=f"cluster_forloop_{root}", graph_attr=dict(style="dotted")) as subg:
            # handle nesting
            to_remove = []
            for remain in todos:
                if remain[0] in children:
                    to_remove.append(remain)
            for each in to_remove:
                todos.remove(each)
            if to_remove:
                draw(subg, to_remove)
            # draw current level
            for each in [root, *children]:
                if each not in drawn:
                    subg.node(str(each))
                    drawn.add(each)


    g = cfg.render_dot()
    while todos:
        draw(g, todos)

    # print(g)
    g.view()



def trace_control(region: List[dis.Instruction], processed) -> BCRegion:
    key = region[0].offset
    if key in processed:
        return processed[key]
    body = []
    edges = []

    result = BCRegion(body=body, edges=edges)
    processed[key] = result

    for i, inst in enumerate(region):
        if i == 0 and inst.opname in SETUP_OPS:
            end = inst.argval
            body.append(inst)
            idx = _find_end_index(region, lambda x: x.offset == end)
            sub = region[i + 1:idx]
            if sub:
                edges.append(trace_control(region[i + 1:], processed))
            if region[idx:]:
                edges.append(trace_control(region[idx:], processed))
            break
        elif i > 0 and (inst.is_jump_target or inst.opname in SETUP_OPS):
            edges.append(trace_control(region[i:], processed))
            break
        elif is_jump(inst):
            if inst.argval > inst.offset:
                # is forward jump
                body.append(inst)
                if is_conditional_jump(inst):
                    edges.append(trace_control(region[i + 1:], processed))
                    edges.append(trace_control(get_body(region, inst.argval), processed))
                else:
                    edges.append(trace_control(get_body(region, inst.argval), processed))
                break
            else:
                # is backedge
                body.append(inst)
                if is_conditional_jump(inst):
                    edges.append(trace_control(region[i + 1:], processed))
                    edges.append(processed[inst.argval])
                else:
                    edges.append(processed[inst.argval])
                break
        else:
            body.append(inst)
    return result


def _find_end_index(sequence, condition):
    for i, x in enumerate(sequence):
        if condition(x):
            return i
    else:
        return None

@dataclass(frozen=True)
class BCRegion:
    body: List[dis.Instruction]
    edges: List[BCRegion]

    def __hash__(self):
        return id(self)

    def __repr__(self):
        # return name_node(self)
        first = self.body[0]
        term = self.terminator
        out = f"BCRegion[{first.offset:06} {first.opname} ... {term.opname}]"
        return out

    @property
    def terminator(self):
        return self.body[-1]

    def dump(self):
        body = self.body
        edges = self.edges
        print(f"BCRegion@{id(self):08x}{{")
        for inst in body:
            print(f"{inst.offset:6}: {inst.opname:20} {inst.argval!r}")
        print(f"#edges: {len(edges)}")
        print("}")

    def show_graphviz(self):
        g = gv.Digraph()
        self._format_graphviz(g, processed=set())
        g.view(filename=f"G_{hex(id(self))}.gv")

    def _format_graphviz(self, g, *, processed):
        if self in processed:
            return
        processed.add(self)

        body = self.body
        edges = self.edges

        cls = self.__class__.__name__
        buf = []
        buf.append(f"{cls}@{id(self):08x}")
        for inst in body:
            buf.append(f"{inst.offset:6}: {inst.opname} ( {inst.argval!r} )")
        buf.append("edges:")
        for edge in edges:
            buf.append(f"{cls}@{id(edge):08x} -> {edge.body[0].offset}")
        buf.append("")
        g.node(str(id(self)), '\l'.join(buf), shape='rect')
        for outregion in edges:
            outregion._format_graphviz(g, processed=processed)
            g.edge(str(id(self)), str(id(outregion)))


def next_offset(body, offset):
    for i, inst in enumerate(body):
        if inst.offset == offset:
            break
    else:
        raise AssertionError("not found")
    return body[i + 1].offset


def get_body(body, start_offset):
    for i, inst in enumerate(body):
        if inst.offset == start_offset:
            break
    else:
        raise AssertionError("not found")
    return body[i:]


def is_jump(inst):
    return inst.opname in JUMP_OPS


def is_conditional_jump(inst):
    return is_jump(inst) and (
        "_IF_" in inst.opname or
        inst.opname in {"FOR_ITER", "SETUP_FINALLY"}
    )


def is_non_condition_jump(inst):
    return is_jump(inst) and not is_conditional_jump(inst)


def is_terminator(inst):
    return inst.opname in TERM_OPS


SETUP_OPS = frozenset({"FOR_ITER", "SETUP_FINALLY", "SETUP_WITH"})
JUMP_OPS = frozenset({dis.opname[x] for x in (dis.hasjabs + dis.hasjrel)})
TERM_OPS = frozenset({"RETURN_VALUE"})
NOP = dis.opmap["NOP"]
