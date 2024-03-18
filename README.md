# elex-solver

This packages includes solvers for:
* Ordinary least squares regression
* Quantile regression
* Transition matrices

## Installation

* We recommend that you set up a virtualenv and activate it (IE ``mkvirtualenv elex-solver`` via http://virtualenvwrapper.readthedocs.io/en/latest/).
* Run ``pip install elex-solver``

## Ordinary least squares
We have our own implementation of ordinary least squares in Python because this let us optimize it towards the bootstrap by storing and re-using the normal equations. This allows for significant speed up.

## Quantile Regression
Since we did not find any implementations of quantile regression in Python that fit our needs, we decided to write one ourselves. At the moment this uses two libraries, the version that solves the non-regularized problem uses `numpy`and solves the dual based on [this](https://arxiv.org/pdf/2305.12616.pdf) paper. The version that solves the regularized problem uses [`cvxpy`](https://www.cvxpy.org/#) and sets up the problem as a normal optimization problem. Eventually, we are planning on replacing the regularized version with the dual also.

## Transition Matrices
We also have a matrix regression solver built with `cvxpy`.  We've used this for our primary election model and analysis.  The transitions it generates form the transitions displayed in our sankey diagrams.

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
