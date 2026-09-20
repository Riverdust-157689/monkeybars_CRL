#!/usr/bin/env bash
# Regenerate / verify patches/jaxgcrl_crl_losses.patch -- the single source of truth
# for our delta against upstream JaxGCRL (see third_party/jaxgcrl/PROVENANCE.md).
#
# Two layouts are supported:
#   * vendored  : third_party/jaxgcrl/ holds the patched sources (tracked, no .git)
#                 and the upstream object database sits in
#                 .scratch/jaxgcrl_upstream_git/  (ignored)
#   * checkout  : third_party/jaxgcrl/ is a real git clone of upstream that has the
#                 patch applied in its working tree
#
# Usage:  tools/make_patch.sh [--check]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TREE="$REPO/third_party/jaxgcrl"
PATCH="$REPO/patches/jaxgcrl_crl_losses.patch"

if [ -d "$TREE/.git" ]; then
    GIT=(git -C "$TREE")
elif [ -d "$REPO/.scratch/jaxgcrl_upstream_git" ]; then
    GIT=(git --git-dir="$REPO/.scratch/jaxgcrl_upstream_git" --work-tree="$TREE")
else
    cat <<'MSG'
no upstream object database found (neither third_party/jaxgcrl/.git nor
.scratch/jaxgcrl_upstream_git).  The vendored tree is still usable at runtime;
to regenerate the patch, re-create the checkout:
  tmp=$(mktemp -d); git clone https://github.com/MichalBortkiewicz/JaxGCRL $tmp/j
  git -C $tmp/j checkout 5a6e7a0
  mv $tmp/j/.git .scratch/jaxgcrl_upstream_git
MSG
    exit 0
fi

current="$(mktemp)"
"${GIT[@]}" diff -- jaxgcrl > "$current"

echo "upstream HEAD : $("${GIT[@]}" rev-parse --short HEAD)"
echo "modified files:"
"${GIT[@]}" diff --name-only -- jaxgcrl | sed 's/^/  /'

if [ "${1:-}" = "--check" ]; then
    if diff -q "$current" "$PATCH" >/dev/null; then
        echo "OK  patches/ is byte-identical to the current working tree delta"
        rm -f "$current"; exit 0
    fi
    echo "FAIL patches/ is STALE -- regenerate (run without --check).  First lines of the drift:"
    diff "$current" "$PATCH" | head -12 || true
    rm -f "$current"; exit 1
fi

mv "$current" "$PATCH"
echo "wrote $PATCH ($(wc -l < "$PATCH") lines)"
diff -q <("${GIT[@]}" diff -- jaxgcrl) "$PATCH" >/dev/null && echo "OK  patch matches the working tree delta"
