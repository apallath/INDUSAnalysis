# INDUSAnalysis

Package to analyze simulation data generated using INDUS.

[Documentation](https://apallath.github.io/INDUSAnalysis) hosted on GitHub pages.

[![build Actions Status](https://github.com/apallath/analysis_scripts/workflows/Analysis/badge.svg)](https://github.com/apallath/analysis_scripts/actions)
[![Website](https://img.shields.io/website?label=docs&url=https%3A%2F%2Fimg.shields.io%2Fwebsite%2Fhttps%2Fapallath.github.io%2FINDUSAnalysis)](https://apallath.github.io/INDUSAnalysis)
[![GitHub license](https://badgen.net/github/license/apallath/INDUSAnalysis)](https://github.com/apallath/INDUSAnalysis/blob/master/LICENSE)

## Installation

1. Install requirements

```sh
pip install -r requirements.txt
```

2. Build C extensions

```sh
python setup.py build_ext --inplace
```

2. Install package [in editable state]

```sh
pip install [-e] .
```

## Tests

Run both unit tests and integration tests to make sure the package is installed
and working correctly. All tests should pass.

Run ensemble tests (scientific tests) to make sure that selected ensemble methods
generate the correct results.

### Unit tests:

Run
```sh
pytest
```
inside the folder `tests_unit/`.

### Integration tests:

Run
```sh
pytest
```
inside the folder `tests_integration/`.

### Ensemble tests:

Run
```sh
pytest
```
inside the folder `tests_ensemble/`.

## Usage

The package can be imported in any Python script using

```python
import INDUSAnalysis
```

INDUSAnalysis library functions can be imported using

```python
import INDUSAnalysis.lib
```

INDUSAnalysis ensemble scripts can be run directly using
```sh
python /path/to/INDUSAnalysis/ensemble/.../script.py
```

## Scripts

`scripts/` contains executable Python and VMD scripts.

Run
```sh
python /path/to/INDUSAnalysis/scripts/run_$analysis_name.py
```
with the required and optional arguments to run analysis on data generated
from a single run [use the -h or --help flags for help with arguments].
