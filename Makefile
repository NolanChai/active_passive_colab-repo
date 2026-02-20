.PHONY: doctor prepare confirmatory raw-sample impulse-sample full-raw full-impulse all-sample all-full

doctor:
	python colab_runner.py --profile doctor --source-mode local

prepare:
	python colab_runner.py --profile prepare --source-mode local

confirmatory:
	python colab_runner.py --profile confirmatory --source-mode local

raw-sample:
	python colab_runner.py --profile raw_signal_sample --source-mode local

impulse-sample:
	python colab_runner.py --profile impulse_sample --source-mode local

full-raw:
	python colab_runner.py --profile full_raw_signal --source-mode local

full-impulse:
	python colab_runner.py --profile full_impulse --source-mode local

all-sample:
	python colab_runner.py --profile all_sample --source-mode local

all-full:
	python colab_runner.py --profile all_full --source-mode local
