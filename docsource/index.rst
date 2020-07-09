.. INDUSAnalysis documentation master file

###############################
INDUSAnalysis
###############################

:Release: |release|

INDUSAnalysis is a Python package to analyze Molecular Dynamics simulation data
for solvated protein systems generated using the GROMACS implementation of
INDUS_.

.. _INDUS: https://link.springer.com/article/10.1007/s10955-011-0269-9

Installation
**************************

**Source code** is available from
`https://github.com/apallath/INDUSAnalysis`_

Obtain the sources with `git`_

.. code-block:: bash

  git clone https://github.com/apallath/INDUSAnalysis.git

.. _`https://github.com/apallath/INDUSAnalysis`: https://github.com/apallath/INDUSAnalysis
.. _git: https://git-scm.com/

1. Install requirements

.. code-block:: bash

  pip install -r requirements.txt

2. Build C extensions

.. code-block:: bash

  python setup.py build_ext --inplace

2. Install package [in editable state]

.. code-block:: bash

  pip install [-e] .

Running tests
**************************

1. For unit tests

.. code-block:: bash

  cd tests_unit
  pytest

1. For integration tests

.. code-block:: bash

  cd tests_integration
  pytest

Documentation
**************************
.. toctree::
  :maxdepth: 2
  :caption: Contents:

  INDUSAnalysis/INDUSAnalysis
  INDUSAnalysis/INDUSAnalysis.lib
