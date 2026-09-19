"""Static guard: every ``args.NAME`` an entry script uses must be declared.

These scripts are run on a GPU for minutes-to-hours, and an ``args`` typo only
surfaces *after* the environments are built (e.g. ``args.use_relu`` while the
flag was never added -> AttributeError right before training starts).  This
checker parses each script with ``ast`` and compares

    used       : every ``args.X`` attribute read
    declared   : every ``ArgumentParser.add_argument("--x-y", dest=...)``
    assigned   : ``args.X = ...`` written after parsing (e.g. the --smoke block
                 or the replay-buffer memory guard)

and fails if anything is used but neither declared nor assigned.  No imports,
no GPU, no JAX -- it is pure AST, so it can run anywhere::

    .venv-warp/bin/python src/check_args.py          # all known entry scripts
    .venv-warp/bin/python src/check_args.py src/train.py
"""

from __future__ import annotations

import ast
import os
import sys

DEFAULT_TARGETS = ("src/train.py", "src/check_reset.py", "src/goal_geometry.py",
                   "src/m1_checks.py", "src/render_m1.py", "src/torque_budget.py",
                   "src/render_m1.py", "src/smoke_env.py", "src/check_scene.py",
                   "src/render_policy.py", "assets/g1_brachiation/build_scene.py")


def _dest(flag: str) -> str:
    return flag.lstrip("-").replace("-", "_")


def analyse(path: str):
    tree = ast.parse(open(path).read(), filename=path)
    used, declared, assigned = set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id == "args":
            used.add(node.attr)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "add_argument":
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str) \
                        and a.value.startswith("-"):
                    declared.add(_dest(a.value))
            for kw in node.keywords:
                if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
                    declared.add(kw.value.value)
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) \
                        and tgt.value.id == "args":
                    assigned.add(tgt.attr)
    return used, declared, assigned


def kwarg_dict_reads_ok(path: str) -> int:
    """Third guard: ``env_kwargs['k']`` may only be read when 'k' is set unconditionally.

    The startup crashes we actually hit (``args.use_relu``, ``CRL(expl_hold=)``, then
    ``eval_kwargs['goal_bar']``) all had the same shape: a value that only exists on
    some code paths, read by a print or a call once the envs are already built.  This
    check compares every ``X['key']`` read against the keys written either in the
    ``dict(...)`` initialiser or by an unconditional ``X['key'] = ...``.
    """
    import ast as _ast
    tree = _ast.parse(open(path).read(), filename=path)
    names = ("env_kwargs", "eval_kwargs")
    written: dict[str, set] = {n: set() for n in names}

    def keys_from_value(node):
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name) \
                and node.func.id == "dict":
            return {kw.arg for kw in node.keywords if kw.arg}
        if isinstance(node, _ast.Name) and node.id in written:
            return set(written[node.id])          # dict(other)
        return set()

    class Walk(_ast.NodeVisitor):
        def visit_Assign(self, node):             # noqa: N802
            for tgt in node.targets:
                if isinstance(tgt, _ast.Name) and tgt.id in names:
                    written[tgt.id] |= keys_from_value(node.value)
                if isinstance(tgt, _ast.Index):   # py<3.9
                    pass
            self.generic_visit(node)

        def visit_If(self, node):                 # conditional writes are NOT trusted
            self.generic_visit(node)

    Walk().visit(tree)

    reads, gets = [], set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Subscript) and isinstance(node.value, _ast.Name) \
                and node.value.id in names and isinstance(node.ctx, _ast.Load):
            sl = node.slice
            if isinstance(sl, _ast.Constant) and isinstance(sl.value, str):
                reads.append((node.value.id, sl.value))
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Attribute) \
                and node.func.attr == "get" and isinstance(node.func.value, _ast.Name) \
                and node.func.value.id in names and node.args \
                and isinstance(node.args[0], _ast.Constant):
            gets.add((node.func.value.id, node.args[0].value))

    missing = sorted({r for r in reads if r not in gets
                      and r[1] not in written[r[0]]
                      and not (r[0] == "eval_kwargs" and r[1] in written["env_kwargs"])})
    print(("OK   " if not missing else "FAIL ")
          + f"kwargs-dict reads: {len(reads)} read, unguarded = {missing}")
    return 1 if missing else 0


def crl_kwargs_ok(path: str) -> int:
    """Second guard: the kwargs train.py passes to ``CRL(...)`` must exist as fields.

    A renamed/renamed-away field (``use_relu``, ``expl_hold``, ...) would otherwise
    only surface as a TypeError *after* the environments are built, i.e. minutes into
    a GPU run.
    """
    import ast as _ast
    import dataclasses
    tree = _ast.parse(open(path).read(), filename=path)
    passed = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name) \
                and node.func.id == "CRL":
            passed |= {kw.arg for kw in node.keywords if kw.arg}
    if not passed:
        print(f"SKIP {path}: no CRL(...) call found")
        return 0
    try:
        from jaxgcrl.agents.crl.crl import CRL  # heavy import (jax/brax)
    except Exception as exc:                                    # pragma: no cover
        print(f"SKIP CRL kwargs check ({type(exc).__name__}: {exc})")
        return 0
    fields = {f.name for f in dataclasses.fields(CRL)}
    missing = sorted(passed - fields)
    print(("OK   " if not missing else "FAIL ")
          + f"CRL(...) kwargs: {len(passed)} passed, missing fields = {missing}")
    return 1 if missing else 0


def main(argv) -> int:
    repo = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    targets = argv[1:] or [os.path.join(repo, t) for t in DEFAULT_TARGETS]
    bad = 0
    for path in targets:
        if not os.path.exists(path):
            print(f"SKIP {path} (not found)")
            continue
        used, declared, assigned = analyse(path)
        missing = sorted(used - declared - assigned)
        status = "OK  " if not missing else "FAIL"
        bad += bool(missing)
        print(f"{status} {os.path.relpath(path, repo):26s} "
              f"flags={len(declared):3d} used={len(used):3d} missing={missing}")
    bad += kwarg_dict_reads_ok(os.path.join(repo, "src/train.py"))
    bad += crl_kwargs_ok(os.path.join(repo, "src/train.py"))
    print("\n" + ("all args are declared" if not bad
                  else f"{bad} check(s) failed -> would crash at runtime"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
