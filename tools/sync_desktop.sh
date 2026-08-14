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

HOST="${MDE_HOST:?Set MDE_HOST to the GPU desktop's SSH user and host}"
KEY="${MDE_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE="${MDE_REMOTE:?Set MDE_REMOTE to the GPU desktop workspace path}"
WORK="$REMOTE/work"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE="${TMPDIR:-/tmp}/mde.bundle"

ssh_() { ssh -i "$KEY" "$HOST" "$@"; }

pull_results() {
  echo "== collecting results from the desktop"
  for d in bench inputs uint8_ab; do
    mkdir -p "$ROOT/reports/$d"
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
  # Always collect first, even for a bare push. The one rule this script exists
  # to enforce is that nothing touches the desktop's tree until its results are
  # here, and a `push` that skipped it was a way to break that rule by hand.
  pull_results

  # ...and then stop if that brought anything new. Collecting is not enough:
  # the bundle is built from HEAD, so a result that arrived a moment ago is not
  # in it, and the `git checkout -- .` below erases it on the desktop while the
  # copy here is still uncommitted and gets overwritten by the next pull. That
  # is how depth_anything_ac's 4.37 ms measurement was lost -- the same class of
  # loss this script was written to prevent, through a door it left open.
  if ! git -C "$ROOT" diff --quiet || ! git -C "$ROOT" diff --cached --quiet; then
    echo "!! results arrived in the pull above and are not committed." >&2
    echo "   commit them, then push again -- the bundle is built from HEAD" >&2
    echo "   and the desktop's copy is about to be reverted." >&2
    git -C "$ROOT" status --short >&2
    exit 1
  fi
  echo "== sending $BRANCH to the desktop"
  git -C "$ROOT" bundle create "$BUNDLE" "$BRANCH" >/dev/null
  scp -i "$KEY" "$BUNDLE" "$HOST:$REMOTE/mde.bundle"
  # git clean under reports/: a result file the desktop wrote is untracked
  # there and tracked in the incoming commit, and git refuses to overwrite it.
  # Removing it is safe only because pull_results just copied it here -- which
  # is why that call is above and not optional.
  ssh_ "cd /d ${WORK//\//\\} && git checkout -- . & git clean -fdq reports & git pull ${REMOTE//\//\\}\\mde.bundle $BRANCH" \
    | tail -3
}

case "${1:-both}" in
  pull) pull_results ;;
  push|both) push_code ;;
  *) echo "usage: $0 [pull|push|both]" >&2; exit 2 ;;
esac
