===========
How to use?
===========

Many commands can be used to create a corpus with the library.
Use the --help function of -h to get more information about the commands.

This is a collection of possible use cases, which we target the library for.

1. Create the vtl corpus with no pre-existing data but not all of the files, i.
   e. only 10_000 samples.

2. Add new data to an already existing corpus, add 10_000 new samples.

3. Filter after words? Create 1000 more "post" word types.



===============
Multiprocessing
===============
The library supports multiprocessing, which can be used to speed up the process and for large corpora this is absolutely necessary.
However it is not enabled by default, to enable it use the --use_mp flag.
For small corpora it is not recommended to use multiprocessing, since generating the melspectrograms is not possible with multiprocessing currently.
The melspectrograms can be generated afterwards too however with the information available in the dataframe, but no solutuion is provided by the library for this yet.

CreateCorpus Class
==========================


.. autoclass:: create_vtl_corpus.create_corpus.CreateCorpus

   :members:


Flags
=====
The following flags can be used to modify the behaviour of the library.

.. argparse::
   :ref: create_vtl_corpus.create_corpus.return_argument_parser
   :prog: create_corpus.py
