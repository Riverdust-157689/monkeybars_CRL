#!/usr/bin/env bash
# Regenerate patches/jaxgcrl_crl_losses.patch from the pinned upstream checkout.
#
# third_party/jaxgcrl is a *shallow clone* of MichalBortkiewicz/JaxGCRL (pinned
# commit, see docs/M2_JaxGCRL接入记录.md).  The root repo ignores third_party/, so
# the only versioned record of our edits to it is this patch file -- which means
# it MUST be regenerated and verified every time jaxgcrl is touched.  (It went
# stale once: the evaluator metric whitelist grew after the patch was written.)
#
# Usage:  tools/make_patch.sh [--check]
#   (no args)  regenerate the patch and verify it reverse-applies to the tree
#   --check    verify only (fails if the patch would not reproduce the tree)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JAXGCRL="$REPO/third_party/jaxgcrl"
PATCH="$REPO/patches/jaxgcrl_crl_losses.patch"

[ -d "$JAXGCRL/.git" ] || { echo "error: $JAXGCRL is not a git checkout"; exit 1; }

if [ "${1:-}" != "--check" ]; then
    git -C "$JAXGCRL" diff > "$PATCH"
    echo "wrote $PATCH ($(wc -l < "$PATCH") lines)"
fi

echo "upstream HEAD : $(git -C "$JAXGCRL" rev-parse --short HEAD)"
echo "modified files:"
git -C "$JAXGCRL" diff --name-only | sed 's/^/  /'

if git -C "$JAXGCRL" apply -R --check "$PATCH" 2>/dev/null; then
    echo "OK  patch reproduces the working tree exactly"
else
    echo "FAIL patch does NOT reverse-apply -- regenerate before committing"
    exit 1
fi
