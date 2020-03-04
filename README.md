# Analysis scripts for MD simulation post processing.

Install in editable state using pip:
`pip install -e .`

## Core analysis package:
`analysis/`

The analysis package can be imported in any Python script using
`import analysis`

The analysis scripts are executable with command line arguments.
`python /path/to/script/ --help`
will list out the required and optional arguments for each script

## Tests:
`test/`

Run tests in this folder by running
`pytest`
inside the folder. Ideally, all tests should pass.

## Other components:
Scratch Jupyter notebooks:
`scratch/'
