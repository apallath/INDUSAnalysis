# INDUSAnalysis

Package to analyze simulation data generated using
[INDUS](https://link.springer.com/article/10.1007/s10955-011-0269-9). 

[Documentation](https://apallath.github.io/INDUSAnalysis) hosted on GitHub pages.

## Status

[![Actions Status](https://img.shields.io/github/workflow/status/apallath/analysis_scripts/Analysis)](https://github.com/apallath/analysis_scripts/actions)
[![Open Issues](https://img.shields.io/github/issues-raw/apallath/analysis_scripts)](https://github.com/apallath/analysis_scripts/issues)
[![Closed Issues](https://img.shields.io/github/issues-closed-raw/apallath/analysis_scripts)](https://github.com/apallath/analysis_scripts/issues)

## Code

[![Python](https://img.shields.io/github/languages/top/apallath/analysis_scripts)](https://www.python.org/downloads/release/python-370/)
[![Google Python Style](https://img.shields.io/badge/Code%20Style-Google%20Python%20Style-brightgreen)](https://google.github.io/styleguide/pyguide.html)

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

`scripts/` contains executable Python and VMD scripts

Run
```sh
python /path/to/INDUSAnalysis/scripts/run_$analysis_name.py
```
with the required and optional arguments to run analysis on data generated
from a single run [use the -h or --help flags for help with arguments].

Run
```sh
python /path/to/INDUSAnalysis/scripts/run_agg_$analysis_name.py
```
with the required and optional arguments to run aggregate analyses on data [
use the -h or --help flags for help with arguments].
