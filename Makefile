# Song Recognition Makefile

.PHONY: help install demo train search test clean

help:
	@echo "🎵 Song Recognition System Commands 🎵"
	@echo "======================================"
	@echo "install     : Install all dependencies"
	@echo "demo        : Run full demo pipeline"
	@echo "train       : Train the model with demo data"
	@echo "search      : Run interactive search demo"
	@echo "test        : Run model performance tests"
	@echo "audio       : Show audio processing guide"
	@echo "clean       : Clean generated files"
	@echo "start       : Run quick start menu"

install:
	pip install -r requirements.txt

demo:
	python demo.py --full

train:
	python train.py --create_demo
	python train.py --data_dir data --epochs 50

search:
	python search.py --interactive

test:
	python search.py --create_test_data
	python search.py --test_performance

audio:
	python demo.py --audio

clean:
	rm -rf data/*.npy
	rm -rf models_saved/*.pth
	rm -rf results/*.png
	rm -rf results/*.json
	rm -rf __pycache__
	rm -rf models/__pycache__
	rm -rf utils/__pycache__

start:
	python start.py
