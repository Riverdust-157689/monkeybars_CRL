#!/usr/bin/env bash
# Reproduce the whole environment for monkeyBars_CRL on a fresh machine.
#
#   git clone <this repo> && cd monkeyBars_CRL && tools/setup_env.sh
#
# Idempotent: safe to re-run.  Needs git, python3.10 (or uv) and network access;
# a GPU is only needed for training (see docs/复现环境.md).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
JAXGCRL_COMMIT=5a6e7a0          # pinned upstream JaxGCRL revision
PY=.venv-warp/bin/python

echo "== 1/5 python 3.10 venv + dependencies =="
if [ ! -x "$PY" ]; then
    if command -v uv >/dev/null 2>&1; then
        uv venv --python 3.10 .venv-warp
    else
        python3.10 -m venv .venv-warp
    fi
fi
if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PY" -r requirements.lock.txt
else
    "$PY" -m pip install -U pip
    "$PY" -m pip install -r requirements.lock.txt
fi

echo "== 2/5 JaxGCRL @ $JAXGCRL_COMMIT + our patch =="
if [ -f third_party/jaxgcrl/jaxgcrl/agents/crl/crl.py ]; then
    echo "   using the VENDORED sources (tracked in this repo; no clone, no patch step)"
    echo "   provenance + re-sync: third_party/jaxgcrl/PROVENANCE.md"
else
    echo "   vendored tree missing -> clone upstream and apply the patch"
    mkdir -p third_party
    git clone https://github.com/MichalBortkiewicz/JaxGCRL third_party/jaxgcrl
    git -C third_party/jaxgcrl checkout "$JAXGCRL_COMMIT"
    git -C third_party/jaxgcrl apply ../../patches/jaxgcrl_crl_losses.patch
fi

echo "== 3/5 Unitree G1 meshes (38 MB, not tracked) =="
if [ ! -f assets/g1_brachiation/menagerie/unitree_g1/g1_with_hands.xml ]; then
    tmp="$(mktemp -d)"
    git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie "$tmp/menagerie"
    mkdir -p assets/g1_brachiation/menagerie
    cp -r "$tmp/menagerie/unitree_g1" assets/g1_brachiation/menagerie/
    rm -rf "$tmp"
fi
( cd assets/g1_brachiation && sha256sum -c menagerie_sha256.txt ) || {
    echo "!! menagerie hashes differ from the ones this repo was built with"
    echo "!! (upstream changed).  Pin an older commit or expect scene changes."; }

echo "== 4/5 rebuild the scene assets from source (pure MuJoCo, no GPU) =="
"$PY" assets/g1_brachiation/build_scene.py --export /tmp/scene_bars_check.xml --spacing 0.40
"$PY" - <<'PY'
import mujoco, numpy as np
a = mujoco.MjModel.from_xml_path("assets/g1_brachiation/scene_bars.xml")
b = mujoco.MjModel.from_xml_path("/tmp/scene_bars_check.xml")
bx = lambda m: np.array([m.body_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"bar{i}")][0] for i in range(5)])
assert (a.nq, a.ngeom) == (b.nq, b.ngeom), (a.nq, a.ngeom, b.nq, b.ngeom)
assert np.allclose(bx(a), bx(b)), (bx(a), bx(b))
print(f"   scene OK: nq={a.nq} ngeom={a.ngeom} bar_x={np.round(bx(a), 4).tolist()}")
PY

echo "== 5/5 static guards (run these before every GPU run) =="
"$PY" src/check_args.py  | tail -1
"$PY" src/check_metrics.py | tail -1
echo
echo "done.  Next: docs/复现环境.md (GPU/Warp notes, run commands, what is not tracked)."
