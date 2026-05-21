# Chess NNUE

NNUE training pipeline for [tonitomc/chess-engine](https://github.com/tonitomc/chess-engine), a UCI-compatible chess engine written in Rust (Built on top of Reckless). This repo contains everything needed to train, validate, and export a neural network evaluation to replace the engine's hand-crafted evaluation (HCE).

## What's in here

- Training configuration for [nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch)
- Dataset notes and sources
- Scripts for checkpointing and exporting `.nnue` files
- Notes on integrating the net back into the engine

## Architecture

Simple 768-input network to start:

```
Input:  768 features (12 piece types × 64 squares)
Hidden: 256 neurons × 2 perspectives (white/black)
Output: 1 scalar (centipawn eval)
```

## Training Data

Using [`nodes5000pv2_UHO.binpack`](https://www.kaggle.com/datasets/joostvandevondele/nodes5000pv2-u-uho) as the base dataset — Stockfish self-play at 5000 nodes per move.

## Training

Requires [nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch) and its dependencies.

```bash
python easy_train.py \
  --training-dataset=data/nodes5000pv2_UHO.binpack \
  --validation-dataset=data/nodes5000pv2_UHO.binpack \
  --num-workers=4 \
  --threads=2 \
  --gpus="0," \
  --batch-size=16384 \
  --max_epoch=400 \
  --network-save-period=10 \
  --random-fen-skipping=3 \
  --start-lambda=1.0 \
  --end-lambda=0.75
```

Checkpoints are saved every 10 epochs as `.ckpt` files and can be resumed with `--resume_from_checkpoint`.

## Exporting

Convert a checkpoint to `.nnue` format for use in the engine:

```bash
python serialize.py last.ckpt net.nnue
```

## Status

- [ ] First training run
- [ ] Basic integration into engine
- [ ] Beat HCE in self-play
- [ ] Retrain with Leela-derived data
