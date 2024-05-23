# Insect Detect - DIY camera trap for automated insect monitoring

<img src="https://raw.githubusercontent.com/maxsitt/insect-detect-docs/main/docs/assets/logo.png" width="500">

[![DOI](https://zenodo.org/badge/580886977.svg)](https://zenodo.org/badge/latestdoi/580886977)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://choosealicense.com/licenses/gpl-3.0/)
[![DOI bioRxiv](https://img.shields.io/badge/bioRxiv-10.1101%2F2023.12.05.570242-B31B1B)](https://doi.org/10.1101/2023.12.05.570242)

# DepthAI SDK

The [DepthAI SDK](https://docs.luxonis.com/projects/sdk/en/latest/) is built on top
of the [DepthAI Python API](https://docs.luxonis.com/projects/api/en/latest/) and
contains classes and functions that can make the development of common tasks easier.

DepthAI SDK is in alpha stage until depthai-sdk 2.0.

---

## Installation

Please make sure that you followed [all steps](https://maxsitt.github.io/insect-detect-docs/software/pisetup/)
to set up your Raspberry Pi before using the OAK-1 camera.

Install the required dependencies for Raspberry Pi + OAK by running:

```
sudo curl -fL https://docs.luxonis.com/install_dependencies.sh | bash
```

Install the package libopenblas-dev (required for latest numpy version):

```
sudo apt install libopenblas-dev
```

Install the python3-venv package:

```
sudo apt install python3-venv
```

Create a virtual environment to avoid dependency and version conflicts of the installed packages:

```
python3 -m venv env_sdk
```

Activate the virtual environment:

```
source env_sdk/bin/activate
```

Update pip:

```
python3 -m pip install --upgrade pip
```

Install the required packages by running:

```
python3 -m pip install -r insect-detect/sdk/requirements_sdk.txt
```

After installing these packages, install the depthai-sdk with:

```
python3 -m pip install --no-dependencies depthai-sdk==1.13.1
```

&nbsp;

You can now test the example SDK scripts, e.g. by running:

```
python3 insect-detect/sdk/sdk_cam_preview.py
```

If you want to deactivate the virtual environment, run:

```
deactivate
```

---

## License

This repository is licensed under the terms of the GNU General Public License v3.0
([GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)).

## Citation

If you use resources from this repository, please cite our paper:

```
Sittinger M, Uhler J, Pink M, Herz A (2024) Insect detect: An open-source DIY camera trap for automated insect monitoring. PLOS ONE 19(4): e0295474. https://doi.org/10.1371/journal.pone.0295474
```
