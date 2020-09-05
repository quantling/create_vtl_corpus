======
Readme
======

.. image:: https://zenodo.org/badge/167427297.svg
   :target: https://zenodo.org/badge/latestdoi/167427297

This package supplies the necessary functions in order to synthesize speech
from a phonemic transscription. Furthermore, it defines helpers to improve the
result if more information as the pitch contour is available.

This python tool is based on the work and on the Matlab code on Yingming Gao.

The overall logic is in ``create_corpus.py`` which executes the approriate functions from top to bottom. The functions are supplied by the other files.


Copyright
=========
As the VocalTractLabAPI.so and the JD2.speaker is GPL v3 the rest of the code
here is GPL as well.  If the code is not dependent on VTL anymore you can use
it under MIT license.

