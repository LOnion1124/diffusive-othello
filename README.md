# Diffusive Othello

A variant of traditional board game [Reversi](https://en.wikipedia.org/wiki/Reversi), supporting both player vs. player and player vs. AI.

## Get Started

Build python environment:

```sh
# create virtual environment
python -m venv venv
# then activate it base on your system
# install torch base on your CUDA version, e.g., CUDA 13.0
pip install torch --index-url https://download.pytorch.org/whl/cu130
# install required packages
pip install -r requirements.txt
```

## How to Play

Play with GUI made by pygame:

```sh
# play PVE (use model in config.yaml)
python gui.py PVE
# play PVP
python gui.py PVP
```

A command-line interface is also provided, but only supports 2-player mode:

```sh
python cli.py
```

## Train Your Own AI
