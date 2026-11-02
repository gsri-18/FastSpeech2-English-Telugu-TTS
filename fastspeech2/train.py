""" FastSpeech 2 training loop (upstream train.py).

[MOD] Called from English/train.py and Telugu/train.py with a single
config.yaml instead of three yaml files. The options are listed in those scripts.
"""

import csv
import datetime
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample, device
from model import FastSpeech2Loss
from dataset import Dataset

from evaluate import evaluate

LOSS_NAMES = ["total", "mel", "mel_postnet", "pitch", "energy", "duration"]
IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30), "IST")


def now():
    """[MOD] Wall-clock time in IST, whatever the machine's time zone (Colab runs in UTC)."""
    return datetime.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")


def pick_amp_dtype(mode):
    """[MOD] Autocast dtype for "fp32", "bf16", "fp16" or "auto". Auto means bf16
    on Apple MPS and Ampere or newer GPUs, and fp16 on older ones such as the T4."""
    if mode in (None, "fp32") or device.type == "cpu":
        return None
    if mode == "auto":
        if device.type == "cuda" and torch.cuda.get_device_capability()[0] < 8:
            return torch.float16
        return torch.bfloat16
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[mode]


def append_csv(path, step, values):
    """[MOD] Loss history as CSV (step, six losses, time) for the loss plots."""
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["step"] + LOSS_NAMES + ["time"])
        w.writerow([step] + ["{:.6f}".format(v) for v in values] + [now()])


def main(args, configs):
    print("[{}] Prepare training ... (device: {})".format(now(), device))

    preprocess_config, model_config, train_config = configs
    # [MOD] fixed seed
    seed = train_config.get("seed", 1234)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    num_workers = getattr(args, "num_workers", 0)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,  # [MOD]
        persistent_workers=num_workers > 0,
    )

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    # [MOD] no nn.DataParallel, training uses one device
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]
    amp_dtype = pick_amp_dtype(getattr(args, "mixed_precision", None)
                               or train_config.get("mixed_precision", "fp32"))
    # [MOD] mixed precision. Only fp16 needs loss scaling, otherwise the scaler does nothing
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype is torch.float16)
    print("autocast dtype: {}".format(amp_dtype))
    if args.restore_step >= total_step:  # [MOD] e.g. the notebook is run again after training finished
        print("[{}] Already trained to step {}, nothing to do".format(now(), args.restore_step))
        return
    print("[{}] Training from step {} to {}".format(now(), args.restore_step, total_step))
    on_checkpoint = getattr(args, "on_checkpoint", None)

    outer_bar = tqdm(total=total_step, desc="Training", position=0, mininterval=30)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1,
                         mininterval=30, disable=args.quiet)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                # [MOD] mixed precision, with weights and optimizer state kept in fp32
                with torch.autocast(device.type, dtype=amp_dtype or torch.float32,
                                    enabled=amp_dtype is not None):
                    # Forward
                    output = model(*(batch[2:]))

                    # Cal Loss
                    losses = Loss(batch, output)
                total_loss = losses[0]

                # Backward
                total_loss = total_loss / grad_acc_step
                scaler.scale(total_loss).backward()
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    # [MOD] unscale first so clipping sees the true gradients
                    scaler.unscale_(optimizer._optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # [MOD] update weights through the scaler (same LR schedule as step_and_update_lr)
                    optimizer._update_learning_rate()
                    scaler.step(optimizer._optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    message1 = "[{}] Step {}/{}, ".format(now(), step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                        *losses
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=losses)
                    append_csv(os.path.join(train_config["path"]["log_path"], "train_losses.csv"), step, losses)

                # [MOD] save before the optional sample logging below
                if step % save_step == 0 or step == total_step:
                    ckpt_file = os.path.join(
                        train_config["path"]["ckpt_path"],
                        "{}.pth.tar".format(step),
                    )
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        ckpt_file,
                    )
                    if on_checkpoint is not None:  # [MOD] e.g. upload to the HF Hub
                        on_checkpoint(ckpt_file, step)

                # [MOD] a failed TensorBoard sample must not stop training
                if step % synth_step == 0:
                    try:
                        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                            batch,
                            output,
                            vocoder,
                            model_config,
                            preprocess_config,
                        )
                        log(
                            train_logger,
                            fig=fig,
                            tag="Training/step_{}_{}".format(step, tag),
                        )
                        sampling_rate = preprocess_config["preprocessing"]["audio"][
                            "sampling_rate"
                        ]
                        log(
                            train_logger,
                            audio=wav_reconstruction,
                            sampling_rate=sampling_rate,
                            tag="Training/step_{}_{}_reconstructed".format(step, tag),
                        )
                        log(
                            train_logger,
                            audio=wav_prediction,
                            sampling_rate=sampling_rate,
                            tag="Training/step_{}_{}_synthesized".format(step, tag),
                        )
                    except Exception as e:
                        outer_bar.write("[warn] sample synthesis at step {} failed: {}".format(step, e))

                if step % val_step == 0:
                    model.eval()
                    message, val_losses = evaluate(model, step, configs, val_logger, vocoder)
                    append_csv(os.path.join(train_config["path"]["log_path"], "val_losses.csv"), step, val_losses)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write("[{}] {}".format(now(), message))

                    model.train()

                if step == total_step:
                    print("[{}] Training finished at step {}".format(now(), step))
                    return
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1

