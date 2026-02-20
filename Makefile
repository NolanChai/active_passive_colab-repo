.PHONY: doctor prepare confirmatory raw-sample impulse-sample full-raw full-impulse

doctor:
	python colab_runner.py --profile doctor

prepare:
	python colab_runner.py --profile prepare

confirmatory:
	python colab_runner.py --profile confirmatory

raw-sample:
	python colab_runner.py --profile raw_signal_sample

impulse-sample:
	python colab_runner.py --profile impulse_sample

full-raw:
	python colab_runner.py --profile full_raw_signal

full-impulse:
	python colab_runner.py --profile full_impulse
