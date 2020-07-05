# INDUSAnalysis

Package to analyze simulation data generated using
[INDUS](https://link.springer.com/article/10.1007/s10955-011-0269-9).
Presently, the focus of the package is on analysis of solvated protein systems.

## Status

[![Actions Status](https://img.shields.io/github/workflow/status/apallath/analysis_scripts/Analysis)](https://github.com/apallath/analysis_scripts/actions)
[![Open Issues](https://img.shields.io/github/issues-raw/apallath/analysis_scripts)](https://github.com/apallath/analysis_scripts/issues)
[![Closed Issues](https://img.shields.io/github/issues-closed-raw/apallath/analysis_scripts)](https://github.com/apallath/analysis_scripts/issues)

## Code

[![Python](https://img.shields.io/github/languages/top/apallath/analysis_scripts)](https://www.python.org/downloads/release/python-370/)
[![Google Python Style](https://img.shields.io/badge/Code%20Style-Google%20Python%20Style-brightgreen)](https://google.github.io/styleguide/pyguide.html)

## Installation

1. Install requirements

`pip install -r requirements.txt`

2. Build C extensions

`python setup.py build_ext --inplace`

2. Install package [in editable state]

`pip install [-e] .`

## Usage

The package can be imported in any Python script using

`import INDUSAnalysis`

INDUSAnalysis library functions can be imported using

`import INDUSAnalysis.lib`

## Scripts

`scripts/` contains executable Python and VMD scripts

Call `python /path/to/INDUSAnalysis/scripts/run_<analysis_name>.py`
with the required and optional arguments to run analysis on data generated from a single run [use -h or --help for help with arguments].

Call `python /path/to/INDUSAnalysis/scripts/run_agg_<analysis_name>.py` with the required and optional arguments to run aggregate analyses on data [use -h or --help for help with arguments].

## Testing

Run both unit tests and integration tests to make sure the package is installed
and working correctly. Ideally, all tests should pass.

### Integration tests:

Run integration tests in this folder by running
`pytest`
inside the folder `tests_integration/`.

### Unit tests:

Run unit tests in this folder by running
`pytest`
inside the folder `tests_unit/`.

## Contributing

[Contributing Guidelines](CONTRIBUTING.md) contain details.
