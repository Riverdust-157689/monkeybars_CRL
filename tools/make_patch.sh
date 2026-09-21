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
# The upstream object database lives only on the machine that created it (it is
# gitignored, 23 MB).  On a fresh clone, --restore-upstream re-creates it from the
# pinned commit so the patch can be regenerated/verified anywhere.
#
# Usage:  tools/make_patch.sh [--check | --restore-upstream]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TREE="$REPO/third_party/jaxgcrl"
PATCH="$REPO/patches/jaxgcrl_crl_losses.patch"
UPSTREAM_URL="https://github.com/MichalBortkiewicz/JaxGCRL"
UPSTREAM_COMMIT="5a6e7a0"      # see third_party/jaxgcrl/PROVENANCE.md
DB="$REPO/.scratch/jaxgcrl_upstream_git"

restore_upstream() {
    mkdir -p "$REPO/.scratch"
    local tmp; tmp="$(mktemp -d)"
    echo "cloning $UPSTREAM_URL @ $UPSTREAM_COMMIT ..."
    git clone --quiet "$UPSTREAM_URL" "$tmp/j"
    git -C "$tmp/j" checkout --quiet "$UPSTREAM_COMMIT"
    rm -rf "$DB"
    mv "$tmp/j/.git" "$DB"
    rm -rf "$tmp"
    echo "OK  upstream object database restored: $DB @ $(git --git-dir="$DB" rev-parse --short HEAD)"
}

if [ "${1:-}" = "--restore-upstream" ]; then
    restore_upstream; exit 0
fi

if [ -d "$TREE/.git" ]; then
    GIT=(git -C "$TREE")
elif [ -d "$REPO/.scratch/jaxgcrl_upstream_git" ]; then
    GIT=(git --git-dir="$REPO/.scratch/jaxgcrl_upstream_git" --work-tree="$TREE")
else
    cat <<MSG
no upstream object database found (neither third_party/jaxgcrl/.git nor
.scratch/jaxgcrl_upstream_git).

Nothing at runtime needs it: the vendored sources (third_party/jaxgcrl/jaxgcrl)
and patches/jaxgcrl_crl_losses.patch are both tracked, so training, eval, render
and the check_* guards work on a fresh clone.  Only *regenerating/verifying* the
patch needs upstream's history -- restore it with:

  tools/make_patch.sh --restore-upstream     # clone $UPSTREAM_URL @ $UPSTREAM_COMMIT
MSG
    if [ "${1:-}" = "--check" ]; then
        echo "FAIL cannot verify patches/ without the upstream object database"
        exit 1
    fi
    echo "FAIL refusing to touch patches/ without the upstream object database"
    exit 2
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
