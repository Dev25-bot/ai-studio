PY ?= .venv/bin/python
UV ?= uv

.PHONY: help setup data run test test-all bench bench-quick docker clean

help:            ## Show available commands
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup:           ## Create .venv (Python 3.11) and install dependencies
	$(UV) venv -p 3.11 .venv
	$(UV) pip install -p $(PY) -r requirements.txt

data:            ## Download small evaluation subsets into data/
	$(PY) data/prepare_data.py --n 100

run:             ## Launch the Gradio app on http://127.0.0.1:7860
	$(PY) app.py

test:            ## Fast unit tests (no model downloads)
	$(PY) -m pytest

test-all:        ## Unit + integration tests (loads small models)
	$(PY) -m pytest -m "integration or not integration"

bench:           ## Full benchmark (all tasks, bench_limit samples)
	$(PY) -m eval.bench

bench-quick:     ## Quick text-only benchmark (5 samples)
	$(PY) -m eval.bench --tasks sentiment summarization next_word --limit 5

docker:          ## Build and run the Docker image (CPU)
	docker build -t ai-studio . && docker run --rm -p 7860:7860 ai-studio

clean:           ## Remove generated outputs and caches
	rm -rf outputs/image_gen outputs/summarization outputs/sentiment outputs/next_word \
	       outputs/translation outputs/ner outputs/qa outputs/image_classify .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
