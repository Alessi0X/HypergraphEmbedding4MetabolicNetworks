# HypergraphEmbedding4MetabolicNetworks

This repository contains the code and data for the paper "Comparing the ability of embedding methods on metabolic hypergraphs for capturing taxonomy-based features".

The paper is currently under review on _Algorithms for Molecular Biology_. A preprint version is available on [bioRxiv](https://doi.org/10.1101/2025.07.10.663860).

## Usage
The `EmbeddingsAndKernels` folder contains separate scripts for computing the embeddings used in the paper. Each script is designed to be run independently, and they can be executed in any order. Each script will load the metabolic pathways dataset (an example dataset is provided in the `data` folder -- more info below) and compute the embeddings or kernels, saving the results in a pickle file.

<!-- ### On running the Edit Kernels
To run the two Edit Kernels it is strongly recommended to compile the core script `levenshteinDistance.pyx` using Cython. This will significantly speed up the computation of the Edit Kernels. To compile the script, create a file named `setup.py` with the following content:

```python
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("levenshteinDistance.pyx", annotate=True, compiler_directives={'language_level' : "3"})
)
```

Then, run the following command in the terminal:

```bash
python setup.py build_ext --inplace
```
This will compile the `.pyx` script into a C/C++ file and then compiles the C/C++ file into an extension module (a `.so` or `.pyd` file on macOS/Linux and Windows, respectively) that can be imported in Python. -->

### On running Graph2Vec
Conversely to the other embedding methods, Graph2Vec loads the dataset where each hypergraph has to be previously converted into its clique projection. An example to do so is provided in the `BagOfNodes/DegreeCentralityEmbedding_refactor.py` script.

### On parallel processing
The following scripts:

- `DegreeCentralityEmbedding_refactor.py`
- `NodeBetweennessEmbedding_refactor.py`
- `PageRankEmbedding_refactor.py`
- `EditKernel_refactor.py`
- `StratifiedEditKernel_refactor.py`
- `JaccardKernel_refactor.py`

support parallel processing to speed up the computation. For those scripts, parallel processing is enabled by default and uses all available CPU cores (see the `num_cores` variable). If you wish to limit the number of cores used, you can modify the `num_cores` variable in the script before running it, or disable it altogether by setting `num_cores = 1`.

AutoEncoders leverage PyTorch's built-in support for GPU acceleration. If a compatible CUDA GPU is available, the code will automatically use it to speed up the training process. If no GPU is available, the code will run on the CPU. macOS users with Apple Silicon chips can also benefit from GPU acceleration using the Metal Performance Shaders (MPS) backend, which is supported by PyTorch: however, they must manually set the `device` variable to `"mps"` in the code.

## Requirements
To run the code, you need to install the following Python packages:
- `torch==2.7.1`
- `torch_geometric==2.6.1`
- `hypergraphx==1.7.7`
- `karateclub==1.2.1`
- `networkx==3.5`
- `scipy==1.15.3`
- `pyclustertend==1.9.0`
- `multiprocess==0.70.18`

The code has been tested with Python 3.12. Preliminary experiments shown compatibility issues with later Python versions (especially with `karateclub` and `pyclustertend`).

## Data
An example of the metabolic pathways dataset used in the paper is available in the file `data/MetabolicPathways_DEMO_DATASET_Python.pkl`. This file contains the metabolic pathways data in a format suitable for analysis. This example dataset is a smaller version of the dataset used in the paper (5 organisms only), and it is intended for demonstration purposes only. The full list of organisms is available as a supplementary file in the paper.

The Pickle file contains a dictionary with `'DATASET'` as the key and a list of dictionaries as the value. Each dictionary in the list represents an organism and contains the following keys:
- `'ID'`: the ID of the organism
- `'simplices_nodelabels'`: the hyperedge list of the organism, where each hyperedge is represented as a _n_-tuple of node labels (strings), with _n_ being the number of nodes in the hyperedge.

## Citation
If you use this code or data in your research, please cite the paper as follows:
```bibtex
@article {Cervellini2025.07.10.663860,
	author = {Cervellini, Mattia and Sinaimeri, Blerina and Matias, Catherine and Martino, Alessio},
	title = {Comparing the ability of embedding methods on metabolic hypergraphs for capturing taxonomy-based features},
	elocation-id = {2025.07.10.663860},
	year = {2025},
	doi = {10.1101/2025.07.10.663860},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/07/15/2025.07.10.663860},
	journal = {bioRxiv}
}
```