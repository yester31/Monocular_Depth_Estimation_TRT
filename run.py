"""Run a model's script in the environment that spec.json says it needs.

    python run.py export unidepth_v2      # ONNX, in the model's own env
    python run.py build  unidepth_v2      # engine + measurement, in trte
    python run.py infer  unidepth_v2      # PyTorch reference, model's env
    python run.py build --all             # every model, each in its own env
    python run.py build --all --dry-run   # print the commands, run nothing

Twelve models, twelve conda environments, and the environment a script needs
depends on the stage rather than the model: exporting ONNX needs the upstream
package, building an engine needs TensorRT, and those deliberately do not live
together. Getting that pairing wrong is not a quick error -- it surfaces as an
ImportError several minutes in, or worse as a silently different torch.

The pairing is already written down in spec.json. This runs it.

What this does not do: the model scripts take no arguments -- encoder, size and
precision are constants edited in the source -- so there is nothing here to
pass through, and inventing flags that quietly rewrite someone's source would
be worse than the cd. `--all` is the only thing added, and it exists because
the twelve-model sweep was being done by hand, one ssh line at a time, which
is how a benchmark run got lost.

Runs anywhere, but only does anything where the environments are.
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import spec as spec_mod  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))

# stage -> (script, which environment). The engine stages run in the shared
# TensorRT environment; everything touching the upstream model runs in its own.
TRT_ENV = "trte"
STAGES = {
    "export": ("onnx_export.py", None),     # None -> the model's own env
    "build": ("onnx2trt.py", TRT_ENV),
    "infer": ("infer.py", None),
}


def conda_envs():
    """name -> prefix, from conda itself.

    These environments live outside conda's envs directory, so `conda run -n`
    does not see them and the prefix is required. The prefix is also
    machine-specific, which is why it is asked for rather than configured.
    """
    exe = os.environ.get("CONDA_EXE") or "conda"
    try:
        out = subprocess.run([exe, "env", "list"], capture_output=True, text=True,
                             timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return {}, exe
    envs = {}
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        prefix = line.split()[-1]
        if os.path.isdir(prefix):
            envs.setdefault(os.path.basename(prefix), prefix)
    return envs, exe


def command(exe, prefix, model, script):
    return [exe, "run", "--no-capture-output", "-p", prefix, "python", script], \
           os.path.join(ROOT, "models", model)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("models", nargs="*")
    ap.add_argument("--all", action="store_true", help="every model with a spec")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-going", action="store_true",
                    help="with --all, continue past a failure")
    args = ap.parse_args()

    specs = spec_mod.load_all()
    wanted = sorted(specs) if args.all else args.models
    if not wanted:
        print("name a model or pass --all. known: " + ", ".join(sorted(specs)))
        return 2
    unknown = [m for m in wanted if m not in specs]
    if unknown:
        print(f"no spec for: {', '.join(unknown)}")
        return 2

    script, fixed_env = STAGES[args.stage]
    envs, exe = conda_envs()
    if not envs and not args.dry_run:
        print("conda found no environments here. This stage has to run where "
              "they are - see tools/sync_desktop.sh.")
        return 1

    failed = []
    for model in wanted:
        env = fixed_env or specs[model]["env"]
        path = os.path.join(ROOT, "models", model, script)
        if not os.path.isfile(path):
            print(f"{model:20} no {script}")
            failed.append(model)
            continue
        prefix = envs.get(env)
        if not prefix and not args.dry_run:
            print(f"{model:20} environment {env!r} not on this machine")
            failed.append(model)
            continue

        # A dry run is the one thing that must work on a machine with none of
        # these environments -- that is where the sweep gets read before it is
        # sent somewhere to run.
        cmd, cwd = command(exe, prefix or f"<prefix of {env}>", model, script)
        # ASCII only: this prints on a cp949 console, where a stray separator
        # character raises inside print and buries the real output.
        print(f"\n=== {model} | {args.stage} | env {env}"
              f"{'' if prefix else '  (not on this machine)'}")
        # flush: the child writes to the same stream unbuffered, so without
        # this the header for each model arrives after that model's output --
        # which in a twelve-model sweep means every heading is off by one.
        print("    " + " ".join(cmd) + f"   (in models/{model})", flush=True)
        if args.dry_run:
            continue
        # PYTHONUTF8: several scripts print characters cp949 cannot encode, and
        # the traceback that produces hides whatever actually went wrong.
        env_vars = dict(os.environ, PYTHONUTF8="1")
        rc = subprocess.run(cmd, cwd=cwd, env=env_vars).returncode
        if rc:
            print(f"    exit {rc}")
            failed.append(model)
            if not args.keep_going and len(wanted) > 1:
                print("    stopping; --keep-going to continue past this")
                break

    if failed:
        print(f"\nfailed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
