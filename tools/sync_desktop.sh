#!/usr/bin/env bash
# Move work between this machine and the GPU box, in the only order that does
# not lose measurements.
#
#     tools/sync_desktop.sh pull     # bring results back from the desktop
#     tools/sync_desktop.sh push     # send committed code to the desktop
#     tools/sync_desktop.sh both     # pull results, then push code
#
# Why this exists: reports/bench/*.json are tracked, and the desktop writes
# them. Running `git checkout -- .` there to clear the way for a pull reverts
# them to the committed versions, discarding whatever was just measured. That
# happened three times by hand -- twice losing a benchmark, once leaving stale
# engine paths in the records. Pull always runs before any git operation on
# the desktop now, and this script is the only thing that should do either.
set -euo pipefail

HOST="${MDE_HOST:-soy@192.168.0.13}"
KEY="${MDE_KEY:-$HOME/.ssh/id_ed25519_codex_soy}"
REMOTE="${MDE_REMOTE:-C:/Users/soy/mde_trt}"
WORK="$REMOTE/work"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE="${TMPDIR:-/tmp}/mde.bundle"

ssh_() { ssh -i "$KEY" "$HOST" "$@"; }

pull_results() {
  echo "== collecting results from the desktop"
  for d in bench inputs; do
    scp -i "$KEY" "$HOST:$WORK/reports/$d/*" "$ROOT/reports/$d/" 2>/dev/null \
      && echo "   reports/$d" || echo "   reports/$d — nothing new"
  done
  for f in accuracy.md accuracy.json; do
    scp -i "$KEY" "$HOST:$WORK/reports/$f" "$ROOT/reports/" 2>/dev/null \
      && echo "   reports/$f" || true
  done
}

push_code() {
  if ! git -C "$ROOT" diff --quiet || ! git -C "$ROOT" diff --cached --quiet; then
    echo "!! uncommitted changes here; the desktop would get a stale tree" >&2
    git -C "$ROOT" status --short >&2
    exit 1
  fi
  echo "== sending $BRANCH to the desktop"
  git -C "$ROOT" bundle create "$BUNDLE" "$BRANCH" >/dev/null
  scp -i "$KEY" "$BUNDLE" "$HOST:$REMOTE/mde.bundle"
  # Safe now: results were collected above, and anything the desktop changed
  # in tracked files is either already here or was never wanted.
  ssh_ "cd /d ${WORK//\//\\} && git checkout -- . & git pull ${REMOTE//\//\\}\\mde.bundle $BRANCH" \
    | tail -3
}

case "${1:-both}" in
  pull) pull_results ;;
  push) push_code ;;
  both) pull_results; push_code ;;
  *) echo "usage: $0 [pull|push|both]" >&2; exit 2 ;;
esac
