.PHONY: help
CONFIG="SMALL"

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install:  ## install training library
	@pip install -q -e .

run: install  ## run training, small model by default
	@train-gpt $(CONFIG)

run_karpathy: install  ## run training, small model by default
	@train-gpt $(CONFIG) --karpathy

lint:  # clean up the source code
	@isort .
	@black . 