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

if [ ! -d "$JAXGCRL/.git" ]; then
    echo "third_party/jaxgcrl is the VENDORED tree (no .git): nothing to regenerate."
    echo "Re-sync procedure (third_party/jaxgcrl/PROVENANCE.md):"
    echo "  tmp=\$(mktemp -d); git clone https://github.com/MichalBortkiewicz/JaxGCRL \$tmp/j"
    echo "  git -C \$tmp/j checkout 5a6e7a0 && git -C \$tmp/j apply \$PWD/patches/jaxgcrl_crl_losses.patch"
    echo "  diff -r --exclude=assets --exclude=__pycache__ \$tmp/j/jaxgcrl third_party/jaxgcrl/jaxgcrl"
    exit 0
fi

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
