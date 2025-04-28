# mGLI-KDA

<div align='center'>
 
<!-- [![preprint](https://img.shields.io/static/v1?label=arXiv&message=2310.12508&color=B31B1B)](https://www.google.com/) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

**Title** - Knot data analysis using multiscale Gauss link integral.

**Authors** - Li Shen, Hongsong Feng, Fengling Li, Fengchun Lei, Jie Wu, and Guo-Wei Wei

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Prerequisites](#prerequisites)
- [Visualization tools](#Visualization-tools)
- [Datasets](#datasets)
- [Modeling with mGLI-based features](#Modeling-with-mGLI-based-features)
    - [mGLI-based B-factor prediction](#i-mgli-based-b-factor-prediction)
    - [Generation of mGLI-based features for protein-ligand complex](#II-Generation-of-mGLI-based-features-for-protein-ligand-complex)
    - [Generation of mGLI-based features for small molecule](#III-Generation-of-mGLI-based-features-for-small-molecule)
    - [Generation of sequence-based features for protein or small molecules](#IV-Generation-of-sequence-based-features-for-protein-or-small-molecules)

- [Results](#results)
    - [I. Modeling the B-factor datasets]()
    - [II. Modeling the PDBbind datasets]()
- [License](#license)
- [Citation](#citation)

---

## Introduction

In the past decade, topological data analysis (TDA) has emerged as a powerful algebraic topology approach in data science. The main technique used in TDA is persistent homology, which tracks topological invariants over the filtration of point cloud data. Although knot theory and related subjects are a focus of study in mathematics, their success in practical applications is quite limited due to the lack of localization and quantization. We address these challenges by introducing knot data analysis (KDA), a new paradigm that incorporates curve segmentation and multiscale analysis into the Gauss link integral.  The resulting multiscale Gauss link integral (mGLI) recovers the global topological properties of knots and links at an appropriate scale and offers a multiscale geometric topology approach to capture the local structures and connectivities in data. The proposed mGLI significantly outperforms other state-of-the-art methods across various benchmark problems in 14  intricately complex biological datasets, including protein flexibility analysis, protein-ligand interactions, hERG potassium channel blockade screening, and quantitative toxicity assessment. Our KDA opens a new research area in data science.

> **Keywords**: Knot data analysis, Gauss link integral, multiscale analysis.

---

## Model Architecture

Schematic illustration of the overall mGLI-based knot data analysis (KDA) platform is shown in below.

![Model Architecture](concepts.png)

Further explain the details in the [paper](https://github.com/WeilabMSU/mGLI-KDA), providing context and additional information about the architecture and its components.

---

## Prerequisites

- numpy                     1.21.0
- scipy                     1.7.3
- pytorch                   1.10.0 
- pytorch-cuda              11.7
- torchvision               0.11.1
- scikit-learn              1.0.2
- python                    3.10.12
- biopandas                 0.4.1
- Biopython                 1.75

---
## Visualization tools

- vispy 0.12.1

- [pyknotid](https://github.com/SPOCKnots/pyknotid)

- pymol

--- 

## Datasets

A brief introduction about the benchmarks.

| Datasets                |Total    | Training Set                 | Test Set                                             |
|-|-----------------------------|------------------------------|------------------------------                        |
| Bfactor-Set364 | 364  [Data](./datasets)     |   -    |      -                                                            |
| Bfactor-small | 30     [Data](./datasets)  |   -    |      -                                                            |
| Bfactor-medium | 36    [Data](./datasets)  |   -    |      -                                                            |
| Bfactor-large | 34   [Data](./datasets)    |   -    |      -                                                            |
| PDBbind-v2007       |1300 |1105  [Label](https://weilab.math.msu.edu/Downloads/mGLI-KDA/PDBbind.zip)                        | 195 [Label](https://weilab.math.msu.edu/Downloads/mGLI-KDA/PDBbind.zip)                         |
| PDBbind-v2013       |2959|2764  [Label](https://weilab.math.msu.edu/Downloads/mGLI-KDA/PDBbind.zip)                        | 195 [Label](https://weilab.math.msu.edu/Downloads/mGLI-KDA/PDBbind.zip)                         |
| PDBbind-v2016       |4057|3767  [Label](https://weilab.math.msu.edu/Downloads/mGLI-KDA/PDBbind.zip)                        | 290 [Label](https://weilab.math.msu.edu/Downloads/mGLI-KDA/PDBbind.zip)                         |
| toxicity: IGC50       |1792 [Data](https://weilab.math.msu.edu/Downloads/toxicity_data.zip)|1434                          | 358 |
| toxicity: LC50       |823 [Data](https://weilab.math.msu.edu/Downloads/toxicity_data.zip)|659 |164
| toxicity: LC50DM       |353 [Data](https://weilab.math.msu.edu/Downloads/mGLI-KDA/toxicity_data.zip)|283                          | 70 
| toxicity: LD50       |7413 [Data](https://weilab.math.msu.edu/Downloads/mGLI-KDA/toxicity_data.zip)                        |5931  | 1482 
| hERG: Zhang data       |1163 [Data](https://weilab.math.msu.edu/Downloads/mGLI-KDA/hERG-data.zip)|927                          | 236
| hERG: Li data       |4813 [Data](https://weilab.math.msu.edu/Downloads/mGLI-KDA/hERG-data.zip)|3721                          | 1092 
| hERG: Cai data       |4447 [Data](https://weilab.math.msu.edu/Downloads/mGLI-KDA/hERG-data.zip)|3954                          | 493 



- PDBbind RawData: the protein-ligand complex structures. Download from [PDBbind database](http://www.pdbbind.org.cn/)
- Label: the .csv file, which contains the protein ID and corresponding binding affinity for PDBbind data.
- Data for B-factor datasets: molecular 3D structures
- Data for hERG and toxicity datasets: molecular 3D structures, SMILES strings, and labels
---

## Modeling with mGLI-based features
### I. mGLI-based B-factor prediction

```shell
# examples, dataset_name options: Bfactor-Set364, Bfactor-large, Bfactor-medium, Bfactor-small
python codes/mGLI-Bfactor.py --dataset_name Bfactor-small
```


### II. Generation of mGLI-based features for protein-ligand complex
Example with PDB 2eg8, generating mGLI features with "bin" manner and "median" statistics for atom-by-atom Gauss linking integral.
"all" can also be used and "std" statistics is also available for atom-by-atom Gauss linking integral, output: 2eg8-complex-median-bin.npy
```shell
python codes/mGLI-protein-ligand.py --pdbid 2eg8 --bin_or_all bin --integral_type median
```

### III. Generation of mGLI-based features for small molecule
Example with the ligand in protein complex PDB 2eg8, generating mGLI features with "bin" manner and "median" statistics for atom-by-atom Gauss linking integral. Statistics of "all" can also be used.  output: 2eg8-ligand-median-bin.npy
```shell
python codes/mGLI-ligand.py --mol2_path datasets/PDBbind/2eg8/2eg8_ligand.mol2 --mol2_id 2eg8 --bin_or_all bin --integral_type median
```
### IV. Generation of sequence-based features for protein or small molecules
Protein sequence embeddings were generated with [Transformer Protein language model](https://github.com/facebookresearch/esm).

Small molecular sequence embeddings were generated with
[Transformer small molecule language model](https://github.com/WeilabMSU/PretrainModels), and [Autoencoder small molecule language model](https://github.com/jrwnter/cddd)

---

## Results

### I. Modeling the Set-364 b-factor dataset
| Models       | [GNM](https://dyn.life.nthu.edu.tw/oGNM/oGNM.php) [result](./Results)  | pfFRI [result](./Results)| ASPH | opFRI [result](./Results)| EH   | mGLI [result](./Results)|
|--------------|-------|-------|------|-------|------|------|
| PCC          | 0.565 | 0.626 | 0.65 | 0.673 | 0.698| 0.762|


### II. Modeling the PDBbind datasets

#### 1. Modeling with \#{mGLI-all & mGLI-lig-all,TF} features
|Datasets                                        | Training Set                  | Test Set| PCC | RMSE (kcal/mol) |
|-------------------------------------------------|-------------                  |---------|-    |-                |
| PDBbind-v2007 [result](./Results)      |1300 |1105  | **0.835** |1.888|
| PDBbind-v2013 [result](./Results)      |2959|2764  | **0.819** |1.930|
| PDBbind-v2016 [result](./Results)      |4057|3767  | 0.857 |1.673|

#### 2. Modeling with \#{mGLI-bin & mGLI-lig-all,TF} features
|Datasets                                        | Training Set                  | Test Set| PCC | RMSE (kcal/mol) |
|-------------------------------------------------|-------------                  |---------|-    |-                |
| PDBbind-v2007 [result](./Results)      |1105| 195  | 0.831 |1.932|
| PDBbind-v2013 [result](./Results)      |2764| 195  | **0.819** | 1.948|
| PDBbind-v2016 [result](./Results)      |3767| 290  | **0.862** |1.671|


Note, twenty gradient boosting regressor tree (GBRT) models were built for each dataset with distinct random seeds such that initialization-related errors can be addressed. The mGLI-based features and transformer-based features were paired with GBRT, respectively. The consensus predictions (\#{mGLI-all & mGLI-lig-all,TF} or \#{mGLI-bin & mGLI-lig-all,TF}) were obtained using predictions from the two types of models. The predictions can be found in the [results](./Results) folder. 

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

- Li Shen, Hongsong Feng, Fengling Li, Fengchun Lei, Jie Wu, and Guo-Wei Wei, "Knot data analysis using multiscale Gauss link integral"

---
