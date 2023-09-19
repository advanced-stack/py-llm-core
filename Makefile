.PHONY: setup


venv:
	@python3 -m venv venv
	-ln -s venv/bin .
	-ln -s venv/bin/activate .

test-setup: venv
	@bin/pip3 install wikipedia

test:
	@bin/pytest

setup: venv
	@bin/pip3 install -U pip
	@bin/pip3 install -e .

