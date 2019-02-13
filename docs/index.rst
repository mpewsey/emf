=================
EMF Documentation
=================

.. image:: https://travis-ci.com/mpewsey/emf.svg?branch=master
    :target: https://travis-ci.com/mpewsey/emf

.. image:: https://readthedocs.org/projects/emf/badge/?version=latest
    :target: https://emf.readthedocs.io/en/latest/?badge=latest

.. image:: https://codecov.io/gh/mpewsey/emf/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/mpewsey/emf

About
=====
This package provides tools for performing electromagnetic field (EMF) analysis
for transmission lines.


Installation
============
The development version of this repository may be installed via pip:

.. code-block:: none

    pip install git+https://github.com/mpewsey/emf#egg=emf


Example
=======
To perform an analysis, simply create a list of phases or phase segments and
pass them to the desired analysis constructor:

.. code-block:: python

    from emf import Phase2D, EMFAnalysis2D

    phases = [
        Phase2D('A', -10, 10.6, 0.033, 525000, 1000, 120, 3, 0.45),
        Phase2D('B', 0, 10.6, 0.033, 525000, 1000, 0, 3, 0.45),
        Phase2D('C', 10, 10.6, 0.033, 525000, 1000, -120, 3, 0.45)
    ]

    emf = EMFAnalysis2D(phases)


Methods can be called on the analysis object to acquire the desired field
values or to generate plots of cross sections.


API Documentation
=================
.. toctree::
    :maxdepth: 2

    emf
    base
