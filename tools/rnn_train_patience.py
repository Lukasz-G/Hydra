"""Run RNNTagger's rnn-train.py / nmt-train.py with an external dev-accuracy
patience cutoff (both default to a fixed 50 epochs with no early stopping).

Both trainers already keep the best-dev-accuracy checkpoint no matter how
long training continues afterward (each tracks the best epoch and only
overwrites its saved parameter file on a new best). This wrapper changes
nothing about WHICH weights get saved — only how long we wait for a plateau
before stopping the clock. A patience-stopped run and a run left going to
its default 50 epochs produce an IDENTICAL checkpoint whenever the true
best epoch precedes the cutoff, which is what the plateau means in the
first place.

Usage:
  python tools/rnn_train_patience.py --patience 10 --min-epochs 15 -- \
      TRAINER_PY_PATH [trainer's own args...]
"""
import re
import subprocess
import sys

args = sys.argv[1:]
patience, min_epochs = 10, 15
if "--patience" in args:
    i = args.index("--patience")
    patience = int(args.pop(i + 1))
    args.pop(i)
if "--min-epochs" in args:
    i = args.index("--min-epochs")
    min_epochs = int(args.pop(i + 1))
    args.pop(i)
if args and args[0] == "--":
    args = args[1:]

cmd = [sys.executable, "-u", *args]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1)
best_acc, best_epoch, seen_epoch = -1.0, 0, 0
saw_any_epoch = False
patience_stopped = False
# rnn-train.py: "18 DevLoss: 76623 DevAccuracy: 82.01"
pat_rnn = re.compile(r"^(\d+)\s+DevLoss:\s+\S+\s+DevAccuracy:\s+([\d.]+)")
# nmt-train.py: "Accuracy: 91.23" (no epoch number printed; count occurrences)
pat_nmt = re.compile(r"^Accuracy:\s+([\d.]+)")
try:
    for line in proc.stdout:
        print(line, end="", flush=True)
        stripped = line.strip()
        m = pat_rnn.match(stripped)
        if m:
            epoch, acc = int(m.group(1)), float(m.group(2))
        else:
            m = pat_nmt.match(stripped)
            if not m:
                continue
            seen_epoch += 1
            epoch, acc = seen_epoch, float(m.group(1))
        saw_any_epoch = True
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
        elif epoch >= min_epochs and (epoch - best_epoch) >= patience:
            print(f"[patience] no dev improvement for {patience} epochs "
                  f"(best {best_acc:.2f} at epoch {best_epoch}); stopping "
                  f"— the saved checkpoint is that best epoch, unaffected "
                  f"by stopping here", flush=True)
            proc.terminate()
            patience_stopped = True
            break
finally:
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

# A patience-stopped run and a naturally-completed run (returncode 0) both
# leave the correct best-dev checkpoint on disk -- either is success. But if
# the wrapped process died on its own (crash, bad args, etc.) before we ever
# asked it to stop, blindly exiting 0 here would mask that failure and let
# the batch driver mark the stage done with nothing actually trained -- which
# is exactly how an nmt-train.py crash (missing --max_*_vocab_size) slipped
# through undetected. Only treat it as success if we caused the stop, the
# process exited cleanly on its own, or it at least logged one real epoch.
if patience_stopped or proc.returncode == 0 or saw_any_epoch:
    sys.exit(0)
else:
    print(f"[rnn_train_patience] wrapped process exited with code "
          f"{proc.returncode} before logging a single epoch -- treating as "
          f"failure", flush=True)
    sys.exit(1)
