# Makefile for installing project dependencies

# Specify the Python version
PYTHON_VERSION := 3.10

# Name of the virtual environment
ENV_NAME := my_env

# Check if the current environment is venv or Conda
ifndef env
    env := venv
endif

# Commands for venv environment
ifeq ($(env),venv)
    # Use venv
    CREATE_ENV_COMMAND := python$(PYTHON_VERSION) -m venv $(ENV_NAME)
    ACTIVATE_ENV_COMMAND := source $(ENV_NAME)/bin/activate
    DEACTIVATE_ENV_COMMAND := deactivate
# Commands for Conda environment
else
    # Use Conda
    CREATE_ENV_COMMAND := conda create --prefix $(ENV_NAME) python=$(PYTHON_VERSION) -y
    ACTIVATE_ENV_COMMAND := conda activate $(ENV_NAME)
    DEACTIVATE_ENV_COMMAND := conda deactivate
endif

# Install project dependencies
install:
    $(MAKE) create_env
    $(MAKE) requirements
    @echo "Installation complete."

# Install dependencies from requirements.txt
requirements:
    @echo "Installing dependencies from requirements.txt..."
    @$(ACTIVATE_ENV_COMMAND) && pip install -r requirements.txt

# Create the virtual environment
create_env:
    @echo "Creating $(env) environment..."
    @$(CREATE_ENV_COMMAND)

# Delete the virtual environment
delete_env:
    @echo "Deleting $(env) environment..."
    @rm -rf $(ENV_NAME)

