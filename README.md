# codebase
Collecting some useful code pieces and best practices

## Folder structure
- Folder structure for a machine learning project with [poetry](./poetry_ml/README.md)

## Git flow and Git branching
- Gitflow

    ![](./images/gitflow.jpeg)
- [Git tutorial](https://github.com/miguelgfierro/codebase/wiki/Git-tutorial?utm_source=substack&utm_medium=email)

## Code style formatter
- [flake8](https://github.com/PyCQA/flake8)
- [Black](https://github.com/psf/black)
- Type hinting and checking with [mypy](https://github.com/python/mypy)

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

## Configuration
- [data classes for Python configuration](https://alexandra-zaharia.github.io/posts/python-configuration-and-dataclasses/)

## Tests with pytest
Pytest is an easy and commonly used testing framework, here is a [tutorial](https://github.com/pluralsight/intro-to-pytest/tree/master)

## Command Line Interface (CLI)
[Click](https://click.palletsprojects.com/en/7.x/) is a Python package for creating beautiful command line interfaces in a composable way with as little code as necessary. It’s the “Command Line Interface Creation Kit”. It’s highly configurable but comes with sensible defaults out of the box. The alternative option is [Typer](https://typer.tiangolo.com/), it is the FastAPI of CLIs and based on Click.

## Semantic Versioning (SemVer)
[SemVar](https://semver.org/) is a popular versioning scheme that is used by a vast amount of open-source projects to communicate the changes included in a version release

## Score production readiness of your ML system
You can measure how ready is your ML system to go to production by scoring 28 actionable tests, you can find it [here](./images/ml-test-score-rubrics-and-scoring.pdf). It is based on the following paper by Google:
- The ML Test Score:
A Rubric for ML Production Readiness and Technical Debt Reduction, [here](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf).



