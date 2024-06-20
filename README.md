# fucciphase

[![License](https://img.shields.io/pypi/l/fucciphase.svg?color=green)](https://github.com/nobias-fht/fucciphase/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/fucciphase.svg?color=green)](https://pypi.org/project/fucciphase)
[![Python Version](https://img.shields.io/pypi/pyversions/fucciphase.svg?color=green)](https://python.org)
[![CI](https://github.com/nobias-fht/fucciphase/actions/workflows/ci.yml/badge.svg)](https://github.com/nobias-fht/fucciphase/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/nobias-fht/fucciphase/branch/main/graph/badge.svg)](https://codecov.io/gh/nobias-fht/fucciphase)

FUCCI cell cycle analysis plugin.

TODO description

## Installation

The best way to run fucciphase is to install it in a virtual conda environment.
Make sure that git installed and can be called from the command line.

(SOON) To install from pip:

```bash
pip install fucciphase
```

If you wish to install it from source:
    
```bash
git clone https://github.com/nobias-fht/fucciphase
cd fucciphase
pip install -e .
```

To use the notebooks, also install jupyter:
    
```bash
pip install jupyter
```


## Usage

Fucci phase currently supports loading a 
[TrackMate](https://imagej.net/plugins/trackmate/) XML file:

```python
from fucciphase import process_trackmate

trackmate_xml = "path/to/trackmate.xml"
channel1 = "MEAN_INTENSITY_CH3"
channel2 = "MEAN_INTENSITY_CH4"

df = process_trackmate(trackmate_xml, channel1, channel2)
print(df["CELL_CYCLE_PERC"])
```

## Development
To develop fucciphase, clone the repository, install fucciphase in your environment
and install the pre-commit hooks:

```bash
git clone https://github.com/nobias-fht/fucciphase
cd fucciphase
pip install -e ".[test, dev]"
pre-commit install
```

If you want to build the documentation, replace the abovementioned pip install by:
```bash
pip install -e ".[test, dev, doc]"
```

## Cite us

(SOON)
