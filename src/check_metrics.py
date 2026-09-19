#!/usr/bin/env python
"""Static guard for the env's coverage metrics / info keys (no JAX tracing).

Three ways this codebase has silently lost diagnostics:

1. a metric is written by ``_coverage`` but not zero-initialised in
   ``_cov_zero_metrics`` (or vice versa) -> the pytree structure of ``metrics``
   differs between ``reset`` and ``step`` and brax's ``EpisodeWrapper`` scan fails,
   or the key silently never appears;
2. an ``info["..."]`` key is read (e.g. a running max/min/counter) but not present
   in ``_cov_zero_info`` -> same class of failure;
3. a metric exists in the env but is missing from JaxGCRL's evaluator whitelist
   (``third_party/jaxgcrl/jaxgcrl/utils/evaluator.py``) -> it never reaches
   ``progress.csv`` / wandb and the run becomes uninterpretable.

Run before every GPU run, next to ``check_args.py``::

    .venv-warp/bin/python src/check_metrics.py
"""
from __future__ import annotations

import ast
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
ENV = os.path.join(REPO, "src", "envs", "brachiation.py")
EVALUATOR = os.path.join(REPO, "third_party", "jaxgcrl", "jaxgcrl", "utils", "evaluator.py")

# JaxGCRL's evaluator hardcodes these five; they are always logged.
CORE = ["reward", "success", "success_easy", "dist", "distance_from_origin"]


def _find(tree: ast.Module, name: str, kind: type) -> ast.AST:
    for node in tree.body:
        if isinstance(node, kind) and getattr(node, "name", None) == name:
            return node
    raise SystemExit(f"FAIL could not find {kind.__name__} {name!r}")


def main() -> int:
    src = open(ENV).read()
    tree = ast.parse(src)
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "Brachiation")
    cov = _find(cls, "_coverage", ast.FunctionDef)

    # --- keys emitted by _coverage's returned metrics dict -------------------
    m_assign = next(n for n in cov.body if isinstance(n, ast.Assign)
                    and getattr(n.targets[0], "id", "") == "m")
    emitted = [k.value for k in m_assign.value.keys]
    # `cov_cross_b1_{tag}` comes from an f-string loop over COVER_RADII in the class.
    radii = re.findall(r"COVER_RADII\s*=\s*\(([^)]*)\)", src)
    if not radii:
        print("FAIL could not read COVER_RADII")
        return 1
    for r in [float(v) for v in radii[0].split(",") if v.strip()]:
        name = f"cov_cross_b1_{int(r * 100):03d}"
        if name not in emitted:
            emitted.append(name)

    # --- keys zero-initialised for metrics / info ---------------------------
    def string_literals(fn_name: str) -> list[str]:
        fn = _find(cls, fn_name, ast.FunctionDef)
        return [n.value for n in ast.walk(fn)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)]

    zero_m = [k for k in string_literals("_cov_zero_metrics") if k.startswith("cov_")]
    zero_i = [k for k in string_literals("_cov_zero_info") if k.startswith("cov_")]
    # `f"cov_cross_b1_{int(r * 100):03d}"` parses as the literal prefix + a formatted
    # part; drop the bare prefix and expand it with the real COVER_RADII names.
    cross = [f"cov_cross_b1_{int(r * 100):03d}"
             for r in [float(v) for v in radii[0].split(",") if v.strip()]]
    zero_m = [k for k in zero_m if k != "cov_cross_b1_"] + cross
    info_reads = sorted({n.slice.value for n in ast.walk(cov)
                         if isinstance(n, ast.Subscript)
                         and isinstance(n.value, ast.Name) and n.value.id == "info"
                         and isinstance(n.slice, ast.Constant)
                         and isinstance(n.slice.value, str)})
    # step() also reads info keys (running counters) -> keep those in the contract
    step = _find(cls, "step", ast.FunctionDef)
    info_reads = sorted(set(info_reads) | {
        n.slice.value for n in ast.walk(step)
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
        and n.value.id == "state.info" and isinstance(n.slice, ast.Constant)
        and isinstance(n.slice.value, str)})

    ev = open(EVALUATOR).read()
    ev_names = set(re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"', ev))

    # `_coverage` also reads running-counter keys indirectly, via the streak()
    # helper: streak(cond, "cov_both_run").  Those literal names are info keys too.
    info_reads = sorted(set(info_reads) | {
        n.value for n in ast.walk(cov)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
        and re.fullmatch(r"cov_[a-z0-9_]*_run", n.value)})

    problems = []
    for k in emitted:
        if k not in zero_m:
            problems.append(f"metric {k!r} emitted by _coverage but not zero-initialised")
        if k not in ev_names:
            problems.append(f"metric {k!r} missing from the evaluator whitelist")
    for k in zero_m:
        if k not in emitted:
            problems.append(f"metric {k!r} zero-initialised but never emitted")
    for k in info_reads:
        if k not in zero_i and not k.startswith(("goal", "dwell", "prev_bar",
                                                 "max_bar", "switches_total")):
            problems.append(f"info[{k!r}] read but not in _cov_zero_info")

    print(f"metrics emitted by _coverage : {len(emitted)}")
    print(f"  {', '.join(sorted(emitted))}")
    print(f"info keys read in reset/step : {len(info_reads)} "
          f"({', '.join(sorted(info_reads))})")
    print(f"core evaluator keys present  : {all(k in ev_names for k in CORE)}")
    if problems:
        for p in problems:
            print(f"FAIL {p}")
        return 1
    print("OK  metric/info keys are consistent and all metrics are logged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
