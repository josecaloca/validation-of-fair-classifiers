# SGH Warsaw School of Economics - Application of Business Intelligence in financial sector analyses
![CI](https://github.com/josecaloca/validation-of-fair-classifiers/actions/workflows/pytest.yml/badge.svg)

This repository contains the materials for the lecture on **Methods to Validate Fair Classifiers** at SGH Warsaw School of Economics, for the course **Application of Business Intelligence in financial sector analyses** [234390-0090].

The repository consists of four main Jupyter notebooks:

1. Exploratory Data Analysis.ipynb
2. Modelling.ipynb
3. Cut-off Agnostic Metrics.ipynb
4. Cut-off Dependent Metrics.ipynb

Additionally, the PowerPoint presentation used during the lecture is included in the repository.

## Requirements

Before getting started, you need to install the UV package manager.
Follow the instructions here: [UV Installation Guide](https://docs.astral.sh/uv/getting-started/installation/).
UV is a fast tool that helps you manage Python versions and install packages easily.


###  Step 1: Download the project
Open a terminal (or command prompt), then run:

```
git clone https://github.com/josecaloca/validation-of-fair-classifiers.git
cd validation-of-fair-classifiers
```
This will download the project to your computer and move you into its folder.

###  Step 2: Set up Python 3.11.5 using `UV`
Run this command to install and use Python version 3.11.5

```
uv python install 3.11.5
```

###  Step 3: Install all required packages
Now, install the libraries the project needs by running:

```
uv sync
```

This will read the `pyproject.toml` file and install everything you need to run the notebooks.


