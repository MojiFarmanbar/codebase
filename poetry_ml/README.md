# Code structure with Poetry, FastAPI, and CLI

## Recommended setup:
Python version management: pyenv/conda  
Package manager: poetry  
Virtual environment: poetry  
Building package: poetry  


## Helpful commands
- Initialize a project: poetry init 
- Initialze a project with a folder called new_project: poetry new new_project  
- Install dependencies: poetry install  
- Add and install a new package: poetry add package_name  
- Run a command inside the virtualenv: poetry run ..  
- Removing a package: poetry remove package_name  
- Activate your virtual environment: poetry shell  
- Create .venv in the project folder: poetry config virtualenvs.in-project true  
- check the config: poetry config  --list  
check [here](https://python-poetry.org/) for more

## Git pre-commit
- pre-commit install (for checking the code style )
- nbstripout --install (to remove output from your notebooks)