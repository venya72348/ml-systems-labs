# ML Systems Labs

Veniamin Vyrovyi, student ID 72348

## Setup

```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Lab 1

```
python lab01/src/print_versions.py
python lab01/src/train_and_measure.py
```

## Labs 6-7 (GPU build of PyTorch)

```
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121
python lab06/src/studies.py all      # Lab 6: see lab06/README.md and lab06/PROPOSAL.md
python lab07/src/run_all.py          # Lab 7: see lab07/README.md
```
