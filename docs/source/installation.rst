Installation
============
This guide is written under the assumption that you use some kind of Debian-based Linux distribution. If you use a different operating system, you may need to adjust the installation instructions accordingly.

First, install conda or mamba and create a conda environment with the dependencies given in the `environment.yml` file.

Installing MFA
==============
Go to `MFA's website <https://montreal-forced-aligner.readthedocs.io/en/latest/getting_started.html>`__ and follow the instructions there. Depending on your architecture, you may need to install the version without CUDA support.

In general, the following command should work

.. code-block:: bash

   conda create -n {language}_aligner -c conda-forge montreal-forced-aligner

We expect you to create environments for each language you want to use. This makes running the aligner in parallel easier.

We used MFA version 3.1.3 for this project. To downgrade MFA to the correct version, run

.. code-block:: bash

   conda install montreal-forced-aligner=3.1.3

Then download the correct dictionary and language model for your language. They must be those for the MFA phoneme set. Currently, we only support English and German. Do this by running the following commands

.. code-block:: bash

   mfa model download dictionary german_mfa
   mfa model download acoustic german_mfa
   # or for English
   mfa model download acoustic english_mfa
   mfa model download dictionary english_mfa

Check that alignment works by testing it on a small dataset. Importantly, do not activate the conda environment before running `create_vtl_corpus`, since the code does that for you, keeping both installations separate.

fastText
========
fastText models are downloaded for you by the `create_vtl_corpus` package.

If you want to use a custom model, rename it to `cc.{language_it_is_for}.300.bin` and place it in the `create_vtl_corpus/create_vtl_corpus/resources` directory.

