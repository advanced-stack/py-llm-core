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
	@bin/pip3 install -U pip build twine
	@bin/pip3 install -e .

# Define the package name and version
PKG_NAME := py-llm-core
MOD_NAME := py_llm_core
PKG_VERSION := 3.1.0

# Define the directory where the package source code is located
PKG_DIR := $(shell pwd)

# Define the directory where the built packages will be stored
BUILD_DIR := dist

# Define the directory where the package source code will be checked out
SRC_DIR := $(PKG_DIR)/src

# Define the URL of the Git repository
GIT_REPO_URL := https://github.com/advanced-stack/$(PKG_NAME).git

# Define the command to run Python
PYTHON := venv/bin/python3


# Define the command to run pip
PIP := pip3

# Define the command to run setuptools
SETUPTOOLS := $(PIP) install --upgrade setuptools

# Define the command to run twine
TWINE := venv/bin/twine

# Define the command to run git
GIT := git

# Define the command to run the build script
BUILD := $(PYTHON) -m build

# Define the command to check the built packages
CHECK := $(TWINE) check $(BUILD_DIR)/*

# Define the target to create a new Git tag
tag:
	@echo "Creating new Git tag $(PKG_VERSION)"
	@$(GIT) tag -a $(PKG_VERSION) -m "Release version $(PKG_VERSION)"

# Define the target to push the Git tag to the remote repository
push:
	@echo "Pushing Git tag $(PKG_VERSION) to remote repository"
	@$(GIT) push --follow-tags

# Define the target to build the package
build:
	@echo "Building package $(PKG_NAME)-$(PKG_VERSION)"
	@$(SETUPTOOLS)
	@$(BUILD)

# Define the target to check the built packages
check:
	@echo "Checking built packages"
	@$(CHECK)

# Define the target to upload the built packages to PyPI
upload:
	@echo "Uploading built packages to PyPI"
	$(TWINE) upload $(BUILD_DIR)/*

display-version:
	echo ${PKG_VERSION}
