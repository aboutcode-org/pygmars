A Simple Python Project Skeleton
================================
This repo attempts to standardize our python repositories using modern python
packaging and configuration techniques. Using this `blog post`_ as inspiration, this
repository will serve as the base for all new python projects and will be adopted to all 
our existing ones as well.

.. _blog post: https://blog.jaraco.com/a-project-skeleton-for-python-projects/

Usage
=====
A brand new project
-------------------
.. code-block:: bash

    git init my-new-repo
    cd my-new-repo
    git pull git@github.com:nexB/skeleton

    # Create the new repo on GitHub, then update your remote
    git remote set-url origin git@github.com:nexB/your-new-repo.git

From here, you can make the appropriate changes to the files for your specific project.

Update an existing project
---------------------------
.. code-block:: bash

    cd my-existing-project
    git remote add skeleton git@github.com:nexB/skeleton
    git fetch skeleton
    git merge skeleton/main --allow-unrelated-histories

This is also the workflow to use when updating the skeleton files in any given repository.


Customizing
-----------

You typically want to perform these customizations:

- remove or update the src/README.rst and tests/README.rst files
- check the configure and configure.bat defaults


Release Notes
-------------

- 2021-05-11: adopt new configure scripts from ScanCode TK that allows correct
  configuration of which Python version is used.
