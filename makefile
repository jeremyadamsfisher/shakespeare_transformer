.PHONY: help
DOCKER_IMG=jeremyadamsfisher1123/shakespeare-gpt:0.0.44
CONDA=micromamba

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

release:  ## bump patch version
	@bump-my-version bump patch
	@git push --tags

lint:  ## clean up the source code
	@isort .
	@black .

build:  ## build the docker image
	@cog build -t $(DOCKER_IMG)

push:  ## push the docker image
	@cog push $(DOCKER_IMG)

run:  ## run something in docker
	@cog run \
		-e PYTHONPATH=. \
		-e TOKENIZERS_PARALLELISM=false \
		-e "WANDB_API_KEY=$$(cat .secrets.json | jq -r .WANDB_API_KEY)" \
		$(OPT)

poke:  ## run interactive docker shell
	@$(MAKE) run OPT=bash

train:  ## run the training program
	@$(MAKE) run OPT="python -O gpt/train.py $(OPT)"

rm_dataset:  ## remove the cached dataset
	@rm -rf wikipedia_ds

pip_freeze: build
	@sed -i 's/requirements.lock/requirements.txt/g' cog.yaml
	@cog run pip freeze > requirements.lock
	@sed -i 's/requirements.txt/requirements.lock/g' cog.yaml