# elex-solver

This packages includes solvers for:
* Quantile regression
* Transition matrices

## Installation

* We recommend that you set up a virtualenv and activate it (IE ``mkvirtualenv elex-solver`` via http://virtualenvwrapper.readthedocs.io/en/latest/).
* Run ``pip install elex-solver``

## Quantile Regression
Since we did not find any implementations of quantile regression in Python that fit our needs, we decided to write one ourselves. This uses [`cvxpy`](https://www.cvxpy.org/#) and sets up quantile regression as a normal optimization problem. We use quantile regression for our election night model.

## Transition matrices
We also have a solver for transition matrices. While this works arbitrarily, we have used this in the past for our primary election night model. We can still use this to create the sankey diagram coefficients.

## Development
We welcome contributions to this repo. Please open a Github issue for any issues or comments you have.

Set up a virtual environment and run:
```
> pip install -r requirements.txt
> pip install -r requirements-dev.txt 
```

## Precommit
To run pre-commit for linting, run:
```
pre-commit run --all-files
```

## Testing
```
> tox
```
