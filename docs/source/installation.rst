==============
Installation
==============
This guide is written under the assumption that you use some kind of Debian based Linux distribution. 
If you use a different operating system, you may need to adjust the installation instructions accordingly.
First install conda or mamba  and create a conda  environment with the dependecies given in the environment.yml file.


Installing MFA
================
Go to `MFAs website <https://montreal-forced-aligner.readthedocs.io/en/latest/getting_started.html>`__
and follow the instructions here. Depending on your architecture you may need to install the version without CUDA support.

Then download the correct dictorary and language model for your language, they must be those for the MFA phoneme set. Currently we only support English and German.


fastText
===========

fastText Models are downloaded for you by the create_vtl_corpus package. 
If you want to use a custom model rename it to  "cc.{language_it_is_for}.300.bin" and place it in the "create_vtl_corpus/create_vtl_corpus/resources directory