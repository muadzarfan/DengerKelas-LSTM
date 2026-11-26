"""Train a small character-level LSTM for Indonesian speech recognition."""

import argparse
import json
import random
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from prepare_data import ROOT, digest, prepare, read_jsonl

SAMPLE_RATE = 16000
N_FFT = 512
HOP = 320
MEL_BINS = 64
SEED = 42
SYMBOLS = " abcdefghijklmnopqrstuvwxyz0123456789"
CHAR_TO_ID = {char: i for i, char in enumerate(SYMBOLS, start=1)}


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalize(text):
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text).split())


def mel_filters():
    frequencies = torch.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
    max_mel = 2595 * np.log10(1 + SAMPLE_RATE / 1400)
    edges = 700 * (10 ** (torch.linspace(0, max_mel, MEL_BINS + 2) / 2595) - 1)
    left, mid, right = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    return torch.minimum((frequencies - left) / (mid - left),
                         (right - frequencies) / (right - mid)).clamp_min(0).T


def load_split(split, audio_root):
    if split not in ("train", "validation"):
        raise ValueError("The LSTM experiment uses train and validation only.")
    manifest = ROOT / f"data/manifests/fleurs_{split}.jsonl"
    filters = mel_filters()
    window = torch.hann_window(400)
    examples = []
    audio_seconds = 0.0
    for row in read_jsonl(manifest):
        path = audio_root / "data/raw/fleurs/audio" / row["archive_member"]
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing. Run with --prepare-data first.")
        audio, rate = sf.read(path, dtype="float32")
        if rate != SAMPLE_RATE or audio.ndim != 1 or len(audio) != row["prepared_num_samples"]:
            raise ValueError(f"Unexpected waveform: {path}")
        if not np.isfinite(audio).all():
            raise ValueError(f"Nonfinite waveform: {path}")
        power = torch.stft(torch.from_numpy(audio), N_FFT, hop_length=HOP,
                           win_length=400, window=window, return_complex=True).abs().square()
        features = (power.T @ filters).clamp_min(1e-6).log()
        features = (features - features.mean(dim=0)) / features.std(dim=0).clamp_min(1e-4)
        reference = normalize(row["training_target"])
        labels = torch.tensor([CHAR_TO_ID[char] for char in reference], dtype=torch.long)
        min_frames = len(labels) + int((labels[1:] == labels[:-1]).sum())
        if not reference or min_frames > len(features):
            raise ValueError(f"Invalid CTC target length: {row['recording_id']}")
        examples.append((features, labels, reference, row["recording_id"]))
        audio_seconds += len(audio) / rate
    print(f"{split}: {len(examples)} recordings, {audio_seconds / 3600:.2f} h", flush=True)
    return examples, audio_seconds


class SpeechLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(MEL_BINS, 128, num_layers=2, batch_first=True, dropout=0.1)
        self.classifier = nn.Linear(128, len(SYMBOLS) + 1)

    def forward(self, features):
        states, _ = self.lstm(features)
        return self.classifier(states).log_softmax(dim=-1)


def batches(examples, size, shuffle=None):
    # Batches contain similar durations to limit padding. The LSTM reads forward,
    # so padding after an utterance cannot affect its earlier predictions.
    ordered = sorted(examples, key=lambda example: len(example[0]))
    groups = [ordered[start:start + size] for start in range(0, len(ordered), size)]
    if shuffle is not None:
        shuffle.shuffle(groups)
    for group in groups:
        features, targets, references, ids = zip(*group)
        yield (pad_sequence(features, batch_first=True), torch.cat(targets),
               torch.tensor([len(x) for x in features]),
               torch.tensor([len(x) for x in targets]), references, ids)


def decode(path):
    chars = []
    previous = None
    for token in path:
        if token != previous and token != 0:
            chars.append(SYMBOLS[token - 1])
        previous = token
    return " ".join("".join(chars).split())


def edit_distance(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for i, expected in enumerate(reference, start=1):
        current = [i]
        for j, predicted in enumerate(prediction, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (expected != predicted)))
        previous = current
    return previous[-1]


@torch.inference_mode()
def evaluate(model, examples, batch_size):
    model.eval()
    rows = []
    for features, _, lengths, _, references, ids in batches(examples, batch_size):
        predictions = model(features).argmax(dim=-1).tolist()
        for tokens, length, reference, recording_id in zip(predictions, lengths, references, ids):
            prediction = decode(tokens[:int(length)])
            words = reference.split()
            ref_chars, pred_chars = reference.replace(" ", ""), prediction.replace(" ", "")
            rows.append({"recording_id": recording_id, "reference": reference,
                         "prediction": prediction, "word_errors": edit_distance(words, prediction.split()),
                         "reference_words": len(words),
                         "character_errors": edit_distance(ref_chars, pred_chars),
                         "reference_characters": len(ref_chars)})
    totals = {key: sum(row[key] for row in rows) for key in
              ("word_errors", "reference_words", "character_errors", "reference_characters")}
    return {**totals, "wer": totals["word_errors"] / totals["reference_words"],
            "cer": totals["character_errors"] / totals["reference_characters"]}, rows


def train(args):
    torch.manual_seed(SEED)
    torch.set_num_threads(args.threads)
    shuffle = random.Random(SEED)
    started = time.perf_counter()
    train_examples, train_seconds = load_split("train", args.audio_root)
    validation, validation_seconds = load_split("validation", args.audio_root)
    model = SpeechLSTM()
    loss_fn = nn.CTCLoss(blank=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    initial, _ = evaluate(model, validation, args.batch_size)
    print(f"Before training: WER {initial['wer']:.2%}, CER {initial['cer']:.2%}", flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    history = []
    best = None
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        loss_sum = 0.0
        for features, labels, lengths, label_lengths, _, _ in batches(train_examples, args.batch_size, shuffle):
            optimizer.zero_grad(set_to_none=True)
            log_probs = model(features).transpose(0, 1)
            loss = loss_fn(log_probs, labels, lengths, label_lengths)
            if not torch.isfinite(loss):
                raise FloatingPointError("CTC training loss is nonfinite.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item() * len(lengths)
        metrics, predictions = evaluate(model, validation, args.batch_size)
        metrics.update(epoch=epoch, train_ctc_loss=loss_sum / len(train_examples),
                       seconds=time.perf_counter() - epoch_started)
        history.append(metrics)
        if best is None or (metrics["wer"], metrics["cer"]) < (best["wer"], best["cer"]):
            best = metrics
            torch.save(model.state_dict(), args.output / "best.pt")
            write_json(args.output / "best_predictions.json", predictions)
        result = {"device": "cpu", "torch": str(torch.__version__), "threads": args.threads, "seed": SEED,
                  "batch_size": args.batch_size, "planned_epochs": args.epochs, "train_examples": len(train_examples),
                  "validation_examples": len(validation), "train_seconds": train_seconds,
                  "validation_seconds": validation_seconds, "parameters": sum(p.numel() for p in model.parameters()),
                  "manifest_sha256": {split: digest(ROOT / f"data/manifests/fleurs_{split}.jsonl")
                                      for split in ("train", "validation")},
                  "initial": initial, "best": best, "epochs": history, "test_evaluated": False,
                  "elapsed_seconds": time.perf_counter() - started}
        write_json(args.output / "metrics.json", result)
        print(json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-data", action="store_true", help="Download and extract train/dev audio, then exit.")
    parser.add_argument("--audio-root", type=Path, default=ROOT, help="Repo directory containing data/raw/fleurs.")
    parser.add_argument("--output", type=Path, default=ROOT / "results/lstm")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1:
        parser.error("epochs, batch size and threads must be positive")
    if args.prepare_data:
        prepare()
    else:
        train(args)


if __name__ == "__main__":
    main()
