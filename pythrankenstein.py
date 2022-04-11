import os
import sys
import ast
import copy
import pickle
import random
from typing import Any
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CodeStats:
    body_node_types: dict = field(default_factory=lambda: defaultdict(set))
    node_examples: dict = field(default_factory=lambda: defaultdict(set))


class ReplaceBody(ast.NodeTransformer):
    def generic_visit(self, node: ast.AST) -> ast.AST:
        candidates = set()
        for fd in node._fields:
            if fd.endswith("body"):
                fv = getattr(node, fd)
                if isinstance(fv, list):
                    candidates.add(fd)

        if candidates:
            cloned = copy.copy(node)
            for fd in candidates:
                setattr(cloned, fd, [ast.Pass()])
            ast.fix_missing_locations(cloned)
            return cloned

        return super().generic_visit(node)


class BuildCodeFreqVisitor(ast.NodeVisitor):
    def __init__(self, stats: CodeStats):
        self.stats = stats

    def generic_visit(self, node: ast.AST) -> Any:
        self._build_stats_node(node)
        return super().generic_visit(node)

    def _build_stats_node(self, node: ast.AST):
        stats = self.stats
        node_body = getattr(node, "body", None)
        if isinstance(node_body, list):
            stats.body_node_types[type(node)].add(tuple(map(type, node_body)))
            # Replace body
            rplbody = ReplaceBody().visit(node)
            serbody = pickle.dumps(rplbody)
            stats.node_examples[type(node)].add(serbody)


class SourceGenerator:
    def __init__(self, stats, *, max_depth=5):
        self.stats = stats
        self._depth = 0
        self._max_depth = max_depth

    def generate(self, node_cls):
        gen_node = self._gen_random_node(node_cls)
        ast.fix_missing_locations(gen_node)
        return gen_node

    def _gen_random_node(self, node_cls):
        self._depth += 1
        try:
            stats = self.stats
            while True:
                body_kinds = random.choice(
                    list(stats.body_node_types[node_cls])
                )

                node_examples = stats.node_examples
                body = []
                for body_node_cls in body_kinds:
                    if body_node_cls in node_examples:
                        if self._depth > self._max_depth:
                            body_node = self._random_pick_example(
                                body_node_cls
                            )
                        else:
                            body_node = self._gen_random_node(body_node_cls)
                        body.append(body_node)

                if body:
                    break

            sel_parent = self._random_pick_example(node_cls)
            sel_parent.body = body
            return sel_parent
        finally:
            self._depth -= 1

    def _random_pick_example(self, node_cls):
        exs = self.stats.node_examples
        return pickle.loads(random.choice(list(exs[node_cls])))


def _iter_py_files(dirpath):
    for base, _, fnlist in os.walk(dirpath):
        for fn in fnlist:
            if fn.endswith(".py"):
                fp = os.path.join(base, fn)
                yield fp


def main():
    try:
        [procname, dirpath] = sys.argv
    except ValueError:
        procname = sys.argv[0]
        print(
            f"""Error: Missing source directory path.
Usage:
    {procname} <directory path>
Example:
    {procname} /path/to/numba/
"""
        )
        sys.exit(1)

    # Read source examples
    stats = CodeStats()
    for fn in _iter_py_files(dirpath):
        with open(fn) as fin:
            body = fin.read()

        tree = ast.parse(body)
        BuildCodeFreqVisitor(stats).visit(tree)

    # Generate source code
    print("-" * 80)
    sgen = SourceGenerator(stats)
    tree = sgen.generate(ast.FunctionDef)
    src = ast.unparse(tree)
    print(src)


if __name__ == "__main__":
    main()
