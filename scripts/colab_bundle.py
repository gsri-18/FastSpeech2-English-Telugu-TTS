""" Pack code + preprocessed features and upload them to a private HF dataset
repo, so a Colab runtime can fetch them (see colab/train_English.ipynb).

    python scripts/colab_bundle.py --repo USER/fs2-iiith-data --lang English
    python scripts/colab_bundle.py --repo USER/fs2-iiith-data --lang Telugu
    python scripts/colab_bundle.py --repo USER/fs2-iiith-data --code_only

Uploads code.zip (the code without data or checkpoints) and
<lang>_preprocessed.zip (everything in <lang>/preprocessed_data). The zips are
written to a temporary directory and stored uncompressed, since float arrays
barely compress anyway.
"""

import argparse
import os
import subprocess
import tempfile
import zipfile

from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", "__pycache__", "downloads", "raw_data", "checkpoints", "logs",
             "preprocessed_data", "tmp", "Results", "Report"}


def zip_code(out):
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for d, dirs, files in os.walk(ROOT):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
            for f in files:
                p = os.path.join(d, f)
                z.write(p, os.path.relpath(p, ROOT))
        # the commit this code comes from, printed by the Colab notebook
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        z.writestr("COMMIT", commit + (" (with uncommitted changes)" if dirty else "") + "\n")
        # small metadata the inference scripts read
        for lang in ("English", "Telugu"):
            for f in ("split.json", "subset_stats.json", "speaker_meta.json"):
                p = os.path.join(ROOT, lang, "raw_data", f)
                if os.path.exists(p):
                    z.write(p, os.path.relpath(p, ROOT))


def zip_features(lang, out):
    src = os.path.join(ROOT, lang, "preprocessed_data")
    assert os.path.exists(os.path.join(src, "train.txt")), "run {}/preprocess.py first".format(lang)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for d, _, files in os.walk(src):
            for f in files:
                p = os.path.join(d, f)
                z.write(p, os.path.relpath(p, ROOT))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--lang", choices=["English", "Telugu"])
    ap.add_argument("--code_only", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        jobs = [("code.zip", zip_code)]
        if args.lang and not args.code_only:
            jobs.append(("{}_preprocessed.zip".format(args.lang),
                         lambda out: zip_features(args.lang, out)))
        for name, fn in jobs:
            out = os.path.join(tmp, name)
            fn(out)
            print("{}: {:.1f} MB, uploading...".format(name, os.path.getsize(out) / 1e6))
            api.upload_file(path_or_fileobj=out, path_in_repo=name,
                            repo_id=args.repo, repo_type="dataset")
    print("done: https://huggingface.co/datasets/{}".format(args.repo))
