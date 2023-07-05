# Craniosource-GAN-PCA-SSM

## Overview

This is a repository to use a GAN, PCA, and statistical shape models to create
[craniosynostosis distance maps](https://github.com/KIT-IBT/cd-map), for
example for the classification of craniosynostosis on purely synthetic data.

![](/assets/graphical_abstract.png)


## Requirements and installation

We used Python version 3.7, but any reasonably recent version of Python3 should
be fine.  Create a virtual environment to install the required packages using
the requirements file: 

``` bash
# Create the virtual environment
mkdir -p $HOME/.venv
python3 -m venv $HOME/.venv/map_creator

# Activate the environment
source $HOME/.venv/map_creator/bin/activate

# Install the dependencies
pip install -r pip_requirements.txt
```

Alternatively, the dependencies are listed below and installing them via `pip`
should suffice to re-create the virtual environment.
- `numpy`
- `scikit-learn`
- `scikit-image`
- `vtk`
- `torch`

## Run demo

The `python` code is cross-platform and should run everywhere.  The demo is a
simple shell script that just calles the script with different input parametern
and should work anywhere where `bash` can be installed.  On Windows, we
recommend WSL.

``` bash
bash ./demo.sh
```

Alternatively, just take a look at the demo file to figure out how to call the
scripts.

## License

All source code is subject to the terms of the General Public License 3.0.
