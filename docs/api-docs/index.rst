SSIP
==================================

SSIP (Safe Surgery by Identifying Pushouts) is a lightweight Python module for automating surgery between qubit CSS (Calderbank-Shor-Steane) codes. SSIP is flexible: it is capable of performing both external surgery, that is surgery between two codeblocks, and internal surgery, that is surgery within the same codeblock. Under the hood, it performs linear algebra governed by universal constructions in the category of chain complexes.

SSIP is ideally suited for codes with blocklengths in the low hundreds, as compute time above this regime can become excessive.

To install, run 

.. code-block::

   pip install ssip

or clone the `Github repository <https://github.com/alexcowtan/SSIP>`_.

Getting started
~~~~~~~~~~~~~~~

SSIP requires Python 3.10 or above, and `Poetry <https://python-poetry.org/docs/#installation>`_.

Apart from installing SSIP itself, to make use of its full functionality you will have to install `GAP <https://www.gap-system.org/>`_, version 4.13.0 or above. This is because SSIP can use the `QDistRnd <https://joss.theoj.org/papers/10.21105/joss.04120>`_ GAP package to compute distances, in the functions ``distance_GAP`` and ``subsystem_distance_GAP``.

Once GAP has been installed, we recommend then adding the GAP directory to ``$PATH``. How best to do this depends on the terminal shell. Alternatively, rather than using GAP in ``$PATH``, distance functions can take the absolute path to the GAP executable as an optional argument.

There is a directory of demo scripts available in the Github repository. For more exhaustive descriptions see the Documentation below.


How to cite
~~~~~~~~~~~

If you wish to cite SSIP in an academic publication, we recommend citing the `software paper <https://arxiv.org/abs/2407.09423>`_. Should you wish, there is also a paper which sets up the theoretical framework used by SSIP, see `here <https://arxiv.org/abs/2301.13738>`_.


User support
~~~~~~~~~~~~

If you find a bug in SSIP, or have a feature request, you can either make an issue on the Github repo or email alexander.cowtan@quantinuum.com.

Documentation
~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst
   :recursive:

   ssip


Indices and tables
~~~~~~~~~~~~~~~~~~

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

