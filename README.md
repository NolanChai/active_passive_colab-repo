# Colab UID Runner

Small runner repo to execute the passive/active UID analysis pipeline on Google Colab.

## What this runs
- Confirmatory stack (fast):
  - `analysis/02_build_pair_tables.py`
  - `analysis/03_confirmatory_tests.py`
  - `analysis/04_dative_style_controls.py`
  - `analysis/06_genre_topic_models.py`
- Raw+signal stack (heavier):
  - `analysis/01_regenerate_raw_uid.py`
  - `analysis/02_build_pair_tables.py`
  - `analysis/05_signal_spike_harmonic.py`
- Propagation stack (heavier):
  - `analysis/07_propagation_impulse.py`

## Colab quickstart
```python
!git clone https://github.com/<your-user>/<this-repo>.git
%cd <this-repo>
!python colab_runner.py --profile doctor
!python colab_runner.py --profile confirmatory
```

## Main options
```bash
python colab_runner.py \
  --source-repo https://github.com/NolanChai/active-passive-alternations.git \
  --source-branch main \
  --profile doctor
```

```bash
python colab_runner.py \
  --source-repo https://github.com/NolanChai/active-passive-alternations.git \
  --source-branch main \
  --profile confirmatory
```

```bash
python colab_runner.py \
  --profile raw_signal_sample \
  --model distilgpt2 \
  --limit-docs 40
```

```bash
python colab_runner.py \
  --profile impulse_sample \
  --model distilgpt2 \
  --limit-docs 8 \
  --k 10
```

## Profiles
- `doctor`: verify source repo has required analysis files/data
- `prepare`: clone source repo + `uv sync` only
- `confirmatory`: run fast confirmatory/model scripts using existing source outputs
- `raw_signal_sample`: regenerate raw traces on a sample, then signal/spike analysis
- `impulse_sample`: run propagation/impulse sample
- `full_raw_signal`: full raw regeneration + pair table + signal metrics
- `full_impulse`: full propagation run

## Output location
Artifacts are written inside the cloned source repo under:
- `analysis/results/`

## Notes
- Colab GPU is used automatically by the source pipeline when available.
- Full runs can still take hours; start with `raw_signal_sample` and `impulse_sample` first.
- This runner does **not** vendor analysis scripts itself. Your `--source-repo/--source-branch` must include the analysis files in `analysis/` and pipeline updates in `src/uid.py`.
