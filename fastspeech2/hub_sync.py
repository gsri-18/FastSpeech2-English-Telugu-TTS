""" Mirror checkpoints and logs to a private Hugging Face Hub repo.

Colab runtimes are temporary, so every saved checkpoint and the loss logs are
uploaded in a background thread as training goes, and a new session resumes
from the newest checkpoint.

Checkpoints are stored as <lang>/checkpoints/<step>.pth.tar and the logs (loss
CSVs, TensorBoard event files, log.txt) under <lang>/logs/.
"""

import os
import re
import shutil

from huggingface_hub import HfApi, hf_hub_download


class HubSync:
    def __init__(self, repo_id, lang, log_dir, create=True):
        self.api = HfApi()
        self.repo_id, self.lang, self.log_dir = repo_id, lang, log_dir
        if create:
            self.api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
        self.futures = []

    def remote_steps(self):
        pat = re.compile(r"^{}/checkpoints/(\d+)\.pth\.tar$".format(re.escape(self.lang)))
        files = self.api.list_repo_files(self.repo_id, repo_type="model")
        return sorted(int(m.group(1)) for m in map(pat.match, files) if m)

    def pull(self, step, ckpt_dir):
        """Download one checkpoint plus the loss CSVs (so the curves continue)."""
        os.makedirs(ckpt_dir, exist_ok=True)
        path = hf_hub_download(self.repo_id, "{}/checkpoints/{}.pth.tar".format(self.lang, step),
                               repo_type="model")
        dst = os.path.join(ckpt_dir, "{}.pth.tar".format(step))
        if not os.path.exists(dst):
            # copy rather than symlink, so the checkpoint survives a cleared HF cache
            shutil.copyfile(path, dst)
        os.makedirs(self.log_dir, exist_ok=True)
        for name in ("train_losses.csv", "val_losses.csv"):
            try:
                src = hf_hub_download(self.repo_id, "{}/logs/{}".format(self.lang, name),
                                      repo_type="model", force_download=True)
            except Exception:
                continue
            with open(src) as fi, open(os.path.join(self.log_dir, name), "w") as fo:
                fo.write(_truncate_csv(fi.read(), step))
        return dst

    def __call__(self, ckpt_file, step):
        """Called by the training loop after each save to upload the checkpoint and logs."""
        self.futures.append(self.api.upload_file(
            path_or_fileobj=ckpt_file,
            path_in_repo="{}/checkpoints/{}".format(self.lang, os.path.basename(ckpt_file)),
            repo_id=self.repo_id, repo_type="model", run_as_future=True,
            commit_message="{} step {}".format(self.lang, step),
        ))
        self.upload_logs(step)
        print("[hub] queued upload of step {} to {}".format(step, self.repo_id))

    def upload_logs(self, step):
        """Upload the loss CSVs, TensorBoard event files and log.txt. Also called
        once after training, since validation at the last step runs after its
        checkpoint was queued."""
        for name in ("train_losses.csv", "val_losses.csv"):
            p = os.path.join(self.log_dir, name)
            if os.path.exists(p):
                with open(p, "rb") as f:  # take a snapshot, the file keeps growing
                    data = f.read()
                self.futures.append(self.api.upload_file(
                    path_or_fileobj=data, path_in_repo="{}/logs/{}".format(self.lang, name),
                    repo_id=self.repo_id, repo_type="model", run_as_future=True,
                    commit_message="{} logs @ {}".format(self.lang, step),
                ))
        # TensorBoard event files (wall-clock time per logged value) + log.txt
        self.futures.append(self.api.upload_folder(
            folder_path=self.log_dir, path_in_repo="{}/logs".format(self.lang),
            repo_id=self.repo_id, repo_type="model", run_as_future=True,
            ignore_patterns=["*.csv"],
            commit_message="{} tensorboard @ {}".format(self.lang, step),
        ))

    def wait(self):
        for f in self.futures:
            f.result()
        self.futures = []


def _truncate_csv(text, step):
    """Drop log rows after the resumed step (they would be repeated), or all rows for a fresh start."""
    lines = text.splitlines()
    keep = [lines[0]] + [l for l in lines[1:] if l and int(l.split(",")[0]) <= step]
    return "\n".join(keep) + "\n"


def setup(args, train_config, lang):
    """Handle --hub_repo / --resume for a per-language train.py.

    With --resume, restore_step is set to the newest checkpoint found locally
    or on the Hub (downloading it if needed)."""
    sync = None
    if args.hub_repo:
        sync = HubSync(args.hub_repo, lang, train_config["path"]["log_path"])
        args.on_checkpoint = sync
    if args.resume:
        ckpt_dir = train_config["path"]["ckpt_path"]
        local = [int(m.group(1)) for m in
                 (re.match(r"(\d+)\.pth\.tar$", f) for f in os.listdir(ckpt_dir)) if m] \
            if os.path.isdir(ckpt_dir) else []
        remote = sync.remote_steps() if sync else []
        best = max(local + remote, default=0)
        if best and best not in local:
            print("[hub] downloading checkpoint {}".format(best))
            sync.pull(best, ckpt_dir)
        args.restore_step = best
        print("resuming from step {}".format(best) if best else "no checkpoint found, starting fresh")
    # a fresh start (step 0) keeps only the header, so committed logs are replaced, not extended
    for name in ("train_losses.csv", "val_losses.csv"):
        p = os.path.join(train_config["path"]["log_path"], name)
        if os.path.exists(p):
            with open(p) as f:
                text = f.read()
            with open(p, "w") as f:
                f.write(_truncate_csv(text, args.restore_step or 0))
    return sync
