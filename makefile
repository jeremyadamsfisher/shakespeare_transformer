.PHONY: help
CONFIG="micro"
DOCKER_IMG=jeremyadamsfisher1123/shakespeare-gpt:$(shell bump-my-version show current_version)

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

bump:  ## bump patch version
	@bump-my-version bump patch

install:  ## install training library
	@pip install -q -e .

run:  ## run training, small model by default
	@train-gpt $(CONFIG)

lint:  # clean up the source code
	@isort .
	@black .

docker_build:
	@docker build -t $(DOCKER_IMG) .
	@docker push $(DOCKER_IMG) 

docker_poke:
	@docker run --rm -ti $(DOCKER_IMG)