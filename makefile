.PHONY: help
CONFIG="micro"
DOCKER_IMG=jeremyadamsfisher1123/shakespeare-gpt:$(shell bump-my-version show current_version)
CONDA=mamba

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

bump:  ## bump patch version
	@bump-my-version bump patch

bootstrap:  ## install training environment
	@$(CONDA) deactivate
	@$(CONDA) remove -n shakespeare --all
	@conda-lock install conda-osx-arm64.lock --name shakespeare

lint:  ## clean up the source code
	@isort .
	@black .

docker_build:  ## build and push the docker image
	@docker build -t $(DOCKER_IMG) .
	@docker push $(DOCKER_IMG) 

docker_poke:  ## run interactive docker shell
	@docker build -t $(DOCKER_IMG) .
	@docker run --rm -ti $(DOCKER_IMG) 

lock:   ## lock the conda env
	@conda-lock lock --kind explicit --micromamba -f env.cuda.yml -f env.yml -p linux-64
	@conda-lock lock --kind explicit --micromamba -f env.yml -p osx-arm64

test:  ## run tests
	@PYTHONPATH=. pytest -vv