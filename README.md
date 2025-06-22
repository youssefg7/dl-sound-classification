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

- Download the data (ESC-50 and UrbanSound8K)

```bash
python tools/download_data.py
```

- Or manually:
  - [ESC-50](https://github.com/karolpiczak/ESC-50): Dataset for Environmental Sound Classification (616 MB)
  - [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html): A Dataset and Taxonomy for Urban Sound Research (5.6 GB)


## Training

- Add a config file for the training
- Run the training

```bash
python train.py --config configs/train/default.yaml
```

## Experiment Tracking

- Run MLflow UI and view the experiments in the browser (http://localhost:5000)

```bash
mlflow ui
```

## Tech Stack

- [uv](https://docs.astral.sh/uv/): For package management.
- [Hydra](https://hydra.cc/): For configuration management.
- [MLflow](https://mlflow.org/): For experiment tracking.
- [PyTorch](https://pytorch.org/): For deep learning, audio processing, and image processing.
- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/): For training and evaluation.