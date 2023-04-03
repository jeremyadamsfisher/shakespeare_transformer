.PHONY: help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install:  ## install training library
	@pip install -q -e .

run: install  ## run training, small model
	@train-gpt

run_large: install   ## run training, large model
	@train-gpt --large