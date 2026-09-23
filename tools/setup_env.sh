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
install_reqs () {   # $1 = requirements file
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "$PY" -r "$1"
    else
        "$PY" -m pip install -r "$1"
    fi
}
if command -v uv >/dev/null 2>&1; then :; else "$PY" -m pip install -U pip; fi
if ! install_reqs requirements.lock.txt; then
    echo "!! the pinned lock did not resolve on this machine -- falling back to the"
    echo "!! curated direct-dependency list (requirements.txt).  Please report which"
    echo "!! package failed so the lock can be fixed."
    install_reqs requirements.txt
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

echo "== 3/5 Unitree G1 meshes (committed with the repo; only fetched if missing) =="
MEN_DIR=assets/g1_brachiation/menagerie
if [ ! -f "$MEN_DIR/unitree_g1/g1_with_hands.xml" ]; then
    mkdir -p "$MEN_DIR"
    src_dir=""
    tmp=""
    if [ -n "${MENAGERIE_DIR:-}" ] && [ -d "${MENAGERIE_DIR}/unitree_g1" ]; then
        echo "   using MENAGERIE_DIR=$MENAGERIE_DIR"
        src_dir="$MENAGERIE_DIR"
    else
        # Only the unitree_g1 subtree is needed, so use a blobless + sparse clone:
        # the official repo is large and a plain `--depth 1` clone downloads the
        # whole tree (that is what tends to die with "curl 56 / TLS connection
        # non-properly terminated" on flaky links).  HTTP/1.1 avoids the HTTP/2
        # mid-pack disconnects, and we retry.
        for attempt in 1 2 3; do
            tmp="$(mktemp -d)"
            log="$tmp/clone.log"
            if git -c http.version=HTTP/1.1 -c http.postBuffer=524288000 \
                   clone --depth 1 --filter=blob:none --sparse \
                   https://github.com/google-deepmind/mujoco_menagerie "$tmp/m" >"$log" 2>&1 \
               && git -C "$tmp/m" sparse-checkout set unitree_g1 >>"$log" 2>&1; then
                src_dir="$tmp/m"; break
            fi
            echo "   clone attempt $attempt/3 failed:"; tail -3 "$log" | sed 's/^/     /'
            rm -rf "$tmp"; tmp=""
        done
    fi
    if [ -n "$src_dir" ]; then
        cp -r "$src_dir/unitree_g1" "$MEN_DIR/"
        echo "   got $MEN_DIR/unitree_g1"
        case "$src_dir" in /tmp/*|/var/tmp/*) rm -rf "$tmp";; esac
    else
        cat <<'MSG'
!! could not fetch the Unitree G1 meshes (network).  Three ways forward:
   1) retry later / on a better link (the clone is retried 3x with HTTP/1.1);
   2) copy them from a machine that already has them (only ~38 MB):

        rsync -av <other-host>:<repo>/assets/g1_brachiation/menagerie/unitree_g1 \
              assets/g1_brachiation/menagerie/
        (cd assets/g1_brachiation && sha256sum -c menagerie_sha256.txt)

   3) point at an existing menagerie checkout and re-run this script:

        MENAGERIE_DIR=/path/to/mujoco_menagerie tools/setup_env.sh
MSG
        exit 1
    fi
fi
# Two different things can drift, with very different consequences:
#   * the MESH subtree  -> changes the physics (the tracked scene_bars*.xml only
#     references these files) => must match, otherwise results are not comparable;
#   * the upstream XMLs -> only used when REBUILDING the scene from source, so a
#     mismatch there is a warning (the tracked scene XML is what actually runs).
ASSETS_HASH_EXPECT=ccd11011746f55d2b2d5dbd4eab63d9c46ce1a4d33b4d9dda5ad7b8e8b053574
assets_hash=$( cd assets/g1_brachiation/menagerie/unitree_g1 \
    && find assets -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d' ' -f1 )
if [ "$assets_hash" != "$ASSETS_HASH_EXPECT" ]; then
    cat <<MSG
!! the MESH subtree differs from the one this repo was built with:
!!   expected $ASSETS_HASH_EXPECT
!!   got      $assets_hash
!! physics may differ -> copy the original meshes (35 MB) from a machine that has them:
!!   rsync -av <host>:<repo>/assets/g1_brachiation/menagerie/unitree_g1/assets/ \
!!         assets/g1_brachiation/menagerie/unitree_g1/assets/
MSG
    exit 1
fi
echo "   mesh subtree OK ($assets_hash)"
if ( cd assets/g1_brachiation && sha256sum -c menagerie_sha256.txt >/dev/null 2>&1 ); then
    echo "   upstream XMLs match too"
else
    echo "   note: the upstream XMLs differ or are absent upstream (upstream may have"
    echo "         renamed them).  They matter only when REBUILDING the scene; the tracked"
    echo "         scene_bars*.xml is what actually runs, so training is unaffected."
fi

# GPU runtime: install the pip CUDA/cuDNN wheels only when the machine has an
# NVIDIA GPU but no system cuDNN (otherwise JAX silently falls back to CPU).
if command -v nvidia-smi >/dev/null 2>&1; then
    if ! "$PY" -c "import jax, sys; sys.exit(0 if any(d.platform=='gpu' for d in jax.devices()) else 1)" >/dev/null 2>&1; then
        echo "   NVIDIA GPU present but jax sees no GPU device -> installing requirements-gpu.txt"
        if command -v uv >/dev/null 2>&1; then
            uv pip install --python "$PY" -r requirements-gpu.txt || true
        else
            "$PY" -m pip install -r requirements-gpu.txt || true
        fi
        "$PY" -c "import jax; print('   jax.devices() =', jax.devices())" || true
    else
        echo "   jax already sees a GPU device"
    fi
fi

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
"$PY" src/check_variants.py | tail -1
# `--deep` is the guard that catches a goal readout that disagrees with step()
# (it compiles a step per variant, ~1 min); the contact controls need the CPU
# platform explicitly so warp does not try to initialise a CUDA device.
JAX_PLATFORMS=cpu "$PY" src/check_contact_sense.py | tail -1
echo
echo "done.  Next: docs/复现环境.md (GPU/Warp notes, run commands, what is not tracked)."
