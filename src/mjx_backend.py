"""MJX backend helper: one place for the Warp/JAX switch and its compat shims.

We take "plan A" for the JaxGCRL integration: brax is used only as a *type*
library (`brax.envs.base.State/Wrapper` and the physics-free env wrappers),
while the physics is driven directly with `mujoco.mjx`:

    mj_model  = mujoco.MjModel.from_xml_path(scene)
    mx_model  = mjx.put_model(mj_model, impl=impl)          # "jax" | "warp"
    mx_data   = mjx.make_data(mj_model, impl=impl, naconmax=..., njmax=...)
    mx_data   = jax.jit(lambda d: mjx.step(mx_model, d))(mx_data)

Two shims are required for the Warp path as of mujoco-mjx 3.13 / warp-lang 1.17:

1. ``WARP_CACHE_PATH`` must point at a writable directory (Warp compiles kernels
   into ``$WARP_CACHE_PATH/<version>``; the default ``~/.cache/warp`` may be
   read-only).
2. ``np.bool`` was removed in numpy >= 1.24 but ``mujoco/mjx/_src/io.py`` still
   references it; the alias must be restored to **``np.bool_``** (not Python
   ``bool``), otherwise numpy bool scalars fall through the type dispatch and
   you get ``Field has_ellipsoid_geom has unsupported type numpy.bool_``.

Neither shim changes physics.  Verified locally: MJX-Warp initialises on the
Warp **CPU** device, so the Warp path can be validated without a GPU (slowly);
throughput numbers must still come from the GPU machine.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DEFAULT_WARP_CACHE = os.path.join(REPO, ".warp-cache")

_WARP_READY = False


def enable_warp_compat(cache_dir: str = DEFAULT_WARP_CACHE) -> bool:
    """Apply the Warp compatibility shims. Returns True if Warp is importable."""
    global _WARP_READY
    os.environ.setdefault("WARP_CACHE_PATH", cache_dir)
    os.makedirs(os.environ["WARP_CACHE_PATH"], exist_ok=True)
    if not hasattr(np, "bool"):          # numpy >= 1.24 removed the alias
        np.bool = np.bool_               # noqa: NPY003  (must be np.bool_, not bool)
    try:
        import warp  # noqa: F401
        _WARP_READY = True
    except Exception:
        _WARP_READY = False
    return _WARP_READY


def warp_devices() -> list:
    import warp as wp
    return [str(d) for d in wp.get_devices()]


def silence_warp_overflow(mx_model, impl: str = "warp"):
    """Stop MJX-Warp printing a block every time a buffer overflows.

    Purely cosmetic (the physics is unchanged), but note the *reason* it matters:
    ``naconmax`` is the contact budget for **all worlds**, so a too-small value
    makes Warp silently drop contacts.  Use :func:`overflow_bits` to check.

    The flag lives on the Warp option object (``opt._impl.warn_overflow``), not on
    MJX's own ``Option`` -- that is why the first version of this helper was a
    silent no-op.
    """
    if impl != "warp":
        return mx_model
    try:
        opt = mx_model.opt
        return mx_model.replace(opt=opt.replace(_impl=opt._impl.replace(warn_overflow=0)))
    except Exception:
        return mx_model


def overflow_bits(data):
    """Per-world Warp overflow bitmask from an ``mjx.Data`` (None if unavailable).

    Bit values (``mujoco_warp._src.types.OverflowType``):
    ``BROADPHASE=4``, ``NARROWPHASE=8``, ``CCD=16``, ``NJMAX_NNZ=2`` ...
    A non-zero value means that world **lost contacts/constraints**, i.e. the
    physics of that step is not trustworthy -- raise ``naconmax``/``njmax``.
    """
    try:
        import numpy as _np
        ov = getattr(getattr(data, "_impl", None), "overflow", None)
        if ov is None:
            return None
        return _np.asarray(ov).reshape(-1)
    except Exception:
        return None


def load_mjx(scene_xml: str, impl: str = "jax", naconmax: Optional[int] = None,
             njmax: Optional[int] = None) -> Tuple[object, object, object]:
    """Return ``(mj_model, mx_model, mx_data)`` for the requested implementation."""
    import mujoco
    import mujoco.mjx as mjx

    if impl == "warp":
        enable_warp_compat()
    mj_model = mujoco.MjModel.from_xml_path(scene_xml)
    mx_model = mjx.put_model(mj_model, impl=impl)
    mx_model = silence_warp_overflow(mx_model, impl)
    kwargs = {}
    if impl == "warp":
        kwargs = dict(naconmax=naconmax or 4096, njmax=njmax or 1024)
    mx_data = mjx.make_data(mj_model, impl=impl, **kwargs)
    return mj_model, mx_model, mx_data


def reset_mjx_data(mj_model, mx_data, key: str = "hang"):
    """Initialise ``mx_data`` with qpos/qvel/ctrl from a MuJoCo keyframe."""
    import jax.numpy as jnp
    import mujoco

    data = mujoco.MjData(mj_model)
    kid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, key)
    mujoco.mj_resetDataKeyframe(mj_model, data, kid)
    return mx_data.replace(qpos=jnp.array(data.qpos.copy()),
                           qvel=jnp.zeros(mj_model.nv),
                           ctrl=jnp.array(data.ctrl.copy()))
