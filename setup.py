from setuptools import setup, find_packages

# read the contents of README file
from os import path

# this_directory = path.abspath(path.dirname(__file__))
# with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
#     long_description = f.read()


setup(
    name="oscar",
    description="cOmpressed Sensing based Cost lAndscape Reconstruction (OSCAR).",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    author="Kun Liu",
    python_requires=">=3, <4",
    packages=["oscar"],
    # install_requires=[
        #   https://mitiq.readthedocs.io/en/stable/changelog.html
        #   "qiskit==0.37.2", 
        #   "mitiq==0.18.0",
    #     "pynauty==1.0.0",
    #     "qiskit-optimization",
    #     "pandas",
    #     "networkx",
    #     "numpy",
    #     "pytest",
    #     "tqdm",
    #     "cvxgraphalgs",
    #     "cvxopt",
    #     "scikit-learn==1.0",
    #     "notebook",
    #     "matplotlib",
    #     "seaborn",
    # ],
    # zip_safe=True,
)
