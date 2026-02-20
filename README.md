# Active/Passive UID Colab Runner

This repository is **self-contained** for Google Colab runs.

It now includes:
- `analysis/` scripts (01-08 + shared `common.py`)
- `src/` pipeline code
- `run_uid_pipeline.py`
- `colab_runner.py` orchestration

No separate source repo clone is required unless you explicitly choose `--source-mode git`.

## Quickstart (Colab)
```python
%cd /content
!git clone https://github.com/NolanChai/active_passive_colab-repo.git
%cd active_passive_colab-repo
```

### 1. Doctor check
```python
!python colab_runner.py --profile doctor --source-mode local
```

### 2. Run everything (sample sanity)
```python
!python colab_runner.py --profile all_sample --source-mode local --model distilgpt2 --limit-docs 40 --k 10
```

### 3. Run everything (full)
```python
!python colab_runner.py --profile all_full --source-mode local --model gpt2 --k 10
```

## Profiles
- `doctor`: verify environment + required files
- `prepare`: install deps and ensure GUM data exists
- `confirmatory`: 02/03/04/06
- `raw_signal_sample`: 01 + 02 + 05 (sample)
- `impulse_sample`: 07 (sample)
- `full_raw_signal`: full raw trace regeneration + signal metrics
- `full_impulse`: full propagation run
- `all_sample`: confirmatory + raw_signal_sample + impulse_sample
- `all_full`: confirmatory + full_raw_signal + full_impulse

## Data behavior
- If `data/en_gum-ud-train.conllu` is missing, `colab_runner.py` auto-downloads it.

## Outputs
Artifacts are written to:
- `analysis/results/`

## Faster reruns
After the first successful dependency install, add:
```bash
--skip-install
```
to profile commands.

## Optional: Git mode
If you do want to run from another repo:
```bash
python colab_runner.py \
  --source-mode git \
  --source-repo https://github.com/<user>/<repo>.git \
  --source-branch <branch> \
  --profile confirmatory
```
