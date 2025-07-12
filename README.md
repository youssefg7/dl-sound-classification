# dl-sound-classification

## Virtual Environment

- Install uv

<details>
  <summary>MacOS/Linux/WSL</summary>

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
</details>
<details>
  <summary>Windows</summary>

  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
</details>

- Create a virtual environment

```bash
uv venv
```

- Activate the virtual environment

<details>
  <summary>MacOS/Linux/WSL</summary>

  ```bash
source .venv/bin/activate
  ```
</details>
<details>
  <summary>Windows</summary>

  ```powershell
.venv\Scripts\activate
  ```
</details>

- Install dependencies

```bash
uv sync
```

## Development

- Activate the post-commit hook for auto-formatting.

```bash
git config core.hooksPath .githooks
```

## Data

1. Download the data (ESC-50 and UrbanSound8K)

```bash
python scripts/download_data.py
```

- Or manually:
  - [ESC-50](https://github.com/karolpiczak/ESC-50): Dataset for Environmental Sound Classification (616 MB)
  - [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html): A Dataset and Taxonomy for Urban Sound Research (5.6 GB)


2. Preprocess the data

```bash
python scripts/prepare_esc50.py
```


## Training

### Quick Start
```bash
# Basic training
python scripts/train.py

# Custom parameters
python scripts/train.py optimizer.lr=0.001 batch_size=64 trainer.max_epochs=50
```

### Hyperparameter Optimization
```bash
# Run Optuna optimization with TPE + Hyperband pruning
python scripts/optimize_hyperparams.py

# Quick test with 5 trials
python scripts/optimize_hyperparams.py optuna.n_trials=5

# Analyze results
python scripts/analyze_study.py --study-name "envnet_esc50_optimization_v1" --plots
```

📖 **For detailed training instructions, hyperparameter optimization, and best practices, see [TRAINING.md](TRAINING.md)**

## Experiment Tracking

- Run MLflow UI and view the experiments in the browser (http://localhost:5000)

```bash
mlflow ui
```

## Experiment Tracking Without Local Hosting

- If you do not wish to host the MLflow locally, you can use grok to visualize to run it on the web.

- To do so, create an .env file in the project directory, then generate an API key from the following website: (https://dashboard.ngrok.com/get-started/your-authtoken)

- Fill the .env file with the following:

```bash
NGROK_AUTHTOKEN=<your-actual-token-here>
```

- Run the MLflow script:

```bash
python scripts/mlflow_ui.py
```

- You’ll see a public URL like https://1234abcd.ngrok.io that anyone can open and see your experiments while you keep the server running.

## Tech Stack

- [uv](https://docs.astral.sh/uv/): For package management.
- [Hydra](https://hydra.cc/): For configuration management.
- [MLflow](https://mlflow.org/): For experiment tracking.
- [PyTorch](https://pytorch.org/): For deep learning, audio processing, and image processing.
- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/): For training and evaluation.