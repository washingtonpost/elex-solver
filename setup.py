import os
from codecs import open

from setuptools import find_packages, setup

INSTALL_REQUIRES = ["cvxpy~=1.5", "numpy~=2.1", "scipy~=1.14"]

THIS_FILE_DIR = os.path.dirname(__file__)

LONG_DESCRIPTION = ""
# Get the long description from the README file
with open(os.path.join(THIS_FILE_DIR, "README.md"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

# The full version, including alpha/beta/rc tags
RELEASE = "2.1.1"
# The short X.Y version
VERSION = ".".join(RELEASE.split(".")[:2])

PROJECT = "elex-solver"
AUTHOR = "The Wapo Newsroom Engineering Team"
COPYRIGHT = "2024, {}".format(AUTHOR)


setup(
    name=PROJECT,
    version=RELEASE,
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.11",
    ],
    description="A package for optimization solvers",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages("src", exclude=["docs", "tests"]),
    package_dir={"": "src"},
    include_package_data=True,
    zip_safe=False,
    install_requires=INSTALL_REQUIRES,
    command_options={
        "build_sphinx": {
            "project": ("setup.py", PROJECT),
            "version": ("setup.py", VERSION),
            "release": ("setup.py", RELEASE),
        }
    },
    py_modules=["elexsolver"],
)
