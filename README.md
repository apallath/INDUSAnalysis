# Analysis package for simulation data obtained using GROMACS implementation of Dynamic INDUS

Uses Python 3.x.

## Installation

1. Install requirements
`pip install -r requirements.txt`

2. Install package [in editable state]
`pip install [--editable] .`

## Usage

The analysis package can be imported in any Python script using

`import analysis`

Individual module scripts can be called directly from the command line with
arguments to process data files.
`python /path/to/script/ --help`
will list out the required and optional arguments for each script.

The meta_analysis package, which implements functions for profling/performance
analysis of the analysis package, can be imported using

`import meta_analysis`

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
