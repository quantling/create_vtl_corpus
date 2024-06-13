======
Readme
======

.. image:: https://zenodo.org/badge/167427297.svg
   :target: https://zenodo.org/badge/latestdoi/167427297

This package supplies the necessary functions in order to synthesize speech
from a phonemic transcription. Furthermore, it defines helpers to improve the
result if more information as the pitch contour is available.

Version 2.0.0 and later
-----------------------
From version 2.0.0 we are relying on the new segment-to-gesture API introduced
in VTL 2.3 and use the JD3.speaker instead of the JD2.speaker.

Old version 1.1.0
-----------------
The original version of this tool is based on the work and on the Matlab code
on Yingming Gao. This can be viewed by checking out the tag ``1.1.0``.

The overall logic is in ``create_corpus.py`` which executes the appropriate
functions from top to bottom. The functions are supplied by the other files.

.. note::

   In the since VTL version 2.3 which can be downloaded as free software from
   https://www.vocaltractlab.de/index.php?page=vocaltractlab-download most of
   the functionality implemented here is available directly from the VTL api.
   Please use the VTL api directly.


Copyright
=========
As the VocalTractLabAPI.so and the JD2.speaker is GPL v3 the rest of the code
here is GPL as well.  If the code is not dependent on VTL anymore you can use
it under MIT license.

Acknowledgments
===============
This research was supported by an ERC advanced Grant (no. 742545), by the
University of Tübingen and by the TU Dresden.

