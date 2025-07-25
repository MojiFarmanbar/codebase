# codebase
Collecting some useful code pieces and best practices

## Folder structure
- Folder structure for a machine learning project with [poetry](./poetry_ml/README.md)

## Git flow and Git branching
- Gitflow

    ![](./images/gitflow.jpeg)
- [Git tutorial](https://github.com/miguelgfierro/codebase/wiki/Git-tutorial?utm_source=substack&utm_medium=email)
- clean up [git branches](https://nickymeuleman.netlify.app/blog/delete-git-branches)

## Code style formatter
- [flake8](https://github.com/PyCQA/flake8)
- [Black](https://github.com/psf/black)
- Type hinting and checking with [mypy](https://github.com/python/mypy)

## Virtual environments and dependency management
- [Python virtual environments](https://www.dataquest.io/blog/a-complete-guide-to-python-virtual-environments/)
- [Conda](https://www.freecodecamp.org/news/why-you-need-python-environments-and-how-to-manage-them-with-conda-85f155f4353c)
- [uv](https://docs.astral.sh/uv/)
- [Poetry](https://realpython.com/dependency-management-python-poetry/), helps you create new projects or maintain existing projects while taking care of dependency management

## Coding guidelines

- [Coding guidelines](https://github.com/recommenders-team/recommenders/wiki/Coding-Guidelines?utm_source=substack&utm_medium=email)
- A list of useful resources:
    - a good [source](https://theaisummer.com/best-practices-deep-learning-code/) of best practices and [github repo](https://github.com/The-AI-Summer/Deep-Learning-In-Production/tree/master/2.%20Writing%20Deep%20Learning%20code:%20Best%20Practises).
    -  [code smells](https://refactoring.guru/refactoring/smells)
    - [python tips](https://book.pythontips.com/en/latest/index.html)
    - [python design patterns](https://github.com/faif/python-patterns)
    - [Using super() in base classes](https://eugeneyan.com/writing/uncommon-python/)
    - Writing [efficient python code](https://www.linkedin.com/posts/youssef-hosni-b2960b135_my-9-kaggle-notebooks-that-will-help-you-activity-7172139063557111808-9KFu/?utm_source=share&utm_medium=member_ios)

## Logging
- [Logging decorators](https://ankitbko.github.io/blog/2021/04/logging-in-python/)
- A deep dive into [python logging](https://medium.com/azure-tutorials/a-deep-dive-into-python-logging-practical-examples-for-developers-ca45a072e709)
- good alternative package for logging, [loguru](https://github.com/Delgan/loguru)

## Pipeline
If you are dealing with functions in a sequential order, where the output of one function serves as the input of the next. Creating pipeline helps in breaking down complex tasks into smaller, more manageable stpes, making code more modular, readbale, and maintainable.

- [Python function](https://samroeca.com/python-function-pipelines.html) pipelines and this [example](https://dzone.com/articles/python-function-pipelines-streamlining-data-proces)
- [Pipeline](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pipe.html) in pandas or spark
- Data transformation pipeline with [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)


## Configuration
- [data classes for Python configuration](https://alexandra-zaharia.github.io/posts/python-configuration-and-dataclasses/)

## Tests with pytest
 - Pytest is an easy and commonly used testing framework, here is a [tutorial](https://github.com/pluralsight/intro-to-pytest/tree/master)
 - aux packages for pytest ![](./images/pytest.png)

## Command Line Interface (CLI)
[Click](https://click.palletsprojects.com/en/7.x/) is a Python package for creating beautiful command line interfaces in a composable way with as little code as necessary. It’s the “Command Line Interface Creation Kit”. It’s highly configurable but comes with sensible defaults out of the box. The alternative option is [Typer](https://typer.tiangolo.com/), it is the FastAPI of CLIs and based on Click.

## Semantic Versioning (SemVer)
[SemVar](https://semver.org/) is a popular versioning scheme that is used by a vast amount of open-source projects to communicate the changes included in a version release

## Production readiness and reproducibility of your ML system
- You can measure how ready is your ML system to go to production by scoring 28 actionable tests, you can find it [here](./images/ml-test-score-rubrics-and-scoring.pdf). It is based on the following paper by Google:
    - The ML Test Score:
A Rubric for ML Production Readiness and Technical Debt Reduction, [here](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf).
- [ML Reproducibility tools and best practices](https://koustuvsinha.com/practices_for_reproducibility/)

## Backend/Frontend application
Sometimes you need to expose the result of your ML solution to end-users by creating a frontend or API. here is some common tools:
- FastAPI using [full stack fastapi template](https://github.com/fastapi/full-stack-fastapi-template)


## MLOps
- [MLOps-Basics](https://github.com/graviraja/MLOps-Basics)
- [Awesome MLOps](https://github.com/visenger/awesome-mlops?utm_source=chatgpt.com)
- [MLOps Coding Course](https://github.com/MLOps-Courses/mlops-coding-course?utm_source=chatgpt.com)
- [Python MLOps Cookbook](https://github.com/noahgift/Python-MLOps-Cookbook)

## ML Engineering

- [Awesome Machine Learning Engineer](https://github.com/superlinear-ai/awesome-machine-learning-engineer)

## Data engineer

- [Data engineer handbook](https://github.com/DataExpert-io/data-engineer-handbook?utm_source=substack&utm_medium=email)

## LLM Engineer tolkit
- [LLM engineer toolkit](https://github.com/KalyanKS-NLP/llm-engineer-toolkit)

## MacBook Pro setup
- [My minimal MacBook Pro setup guide](https://eugeneyan.com/writing/mac-setup/?utm_source=convertkit&utm_medium=email&utm_campaign=My%20Minimal%20MacBook%20Pro%20Setup%20Guide%20-%2015710855#macos-settings)
- [Setup Latex in VSCode](https://mathjiajia.github.io/vscode-and-latex/)
