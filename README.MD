# CAML

<div align='center'>
 
<!-- [![preprint](https://img.shields.io/static/v1?label=arXiv&message=2310.12508&color=B31B1B)](https://www.google.com/) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

**Title** - CAML: Commutative algebra machine learning --- A case study on protein-ligand binding affinity prediction.

**Authors** - Hongsong Feng, Faisal Suwayyid, Mushal Zia, JunJie Wee, Yuta Hozumi, Chunlong Chen, and Guo-Wei Wei.

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Prerequisites](#prerequisites)
- [Datasets](#datasets)
- [Modeling with PSRT-based features](#Modeling-with-PSRT-based-features)
    - [Generation of PSRT-based features for protein-ligand complex](#II-Generation-of-PSRT-based-features-for-protein-ligand-complex)

- [Results](#results)
    - [II. Modeling the PDBbind datasets]()
- [License](#license)
- [Citation](#citation)

---

## Introduction

Recently, Suwayyid and Wei have introduced commutative algebra as an emerging paradigm for machine learning and data science. In this work, we propose commutative algebra machine learning (CAML) for the prediction of protein-ligand binding affinities. Specifically, we apply persistent Stanley–Reisner theory, a key concept in combinatorial commutative algebra, to the affinity predictions of protein-ligand binding and metalloprotein-ligand binding. We present three new algorithms, i.e., element-specific commutative algebra, category-specific commutative algebra, and commutative algebra on bipartite complexes, to tackle the complexity of data involved in (metallo) protein-ligand complexes. We show that the proposed CAML outperforms other state-of-the-art methods in (metallo) protein-ligand binding affinity predictions, indicating the great potential of commutative algebra learning.   

> **Keywords**: Persistent commutative algebra, facet persistence barcodes, persistent ideals, commutative algebra learning, protein-ligand binding.

---

## Model Architecture

An illustration of the filtration process of the persistent commutative algebra is shown below.

![Model Architecture](scheme.png)

Further explain the details in the [paper](https://github.com/WeilabMSU/CAML), providing context and additional information about the architecture and its components.

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
--- 

## Datasets

A brief introduction about the benchmark datasets.

| Datasets                |Total    | Training Set                 | Test Set                                             |
|-|-----------------------------|------------------------------|------------------------------                        |
| PDBbind-v2016       |4057 [data](./dataset)|3767  [data](./dataset)                        | 290 [data](./dataset)                         |
| Metalloprotein-ligand       |2463 [data](./dataset)|1845  [data](https://weilab.math.msu.edu/Downloads/PSRT/PDBbind.zip)                        | 618 [data](./dataset)                         |


- PDBbind-v2016: the protein-ligand complex structures. Download from [PDBbind database](http://www.pdbbind.org.cn/)
- Metalloprotein-ligand: the metalloprotein-ligand complex structures were complied from PDBbind-v2020 ([PDBbind database](http://www.pdbbind.org.cn/)) by [Jiang2023]
- data: the .csv file, which contains the protein ID and corresponding binding affinity for PDBbind data.
---

## Modeling with PSRT-based features

### I. Build machine learning models using PSRT-based features.
```shell
python src/build_model.py --dataset_name v2016
```
### II. Generation of sequence-based features for protein and small molecules
Protein sequence embeddings were generated with [Transformer Protein language model ESM2](https://github.com/facebookresearch/esm) [Rives2021].

Small molecular sequence embeddings were generated with [Transformer small molecule language model](https://github.com/WeilabMSU/PretrainModels) [Chen2021]. The input small molecular sequence is SMILES string. Instructions on molecular descriptors using this language model is provided in the github.

## Results

### I. Modeling the (Metallo)protein-ligand datasets

|Datasets                                        | Training Set                  | Test Set| PCC | RMSE (kcal/mol) |
|-------------------------------------------------|-------------                  |---------|-    |-                |
| PDBbind-v2016 [result](./Results)      |  3767 | 290 | 0.858 |  1.673|
| Metalloprotein-ligand [result](./Results) |1845| 618 | 0.745/0.755 |  1.947|


Note, twenty gradient boosting regressor tree (GBRT) models were built for each dataset with 20 indenpedent runs with different random numbers. The PSRT-based features and transformer-based features were paired with GBRT, respectively. The predictions can be found in the [results](./Results) folder. 

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

- [Jiang2023] Dejun Jiang, Zhaofeng Ye, Chang-Yu Hsieh, Ziyi Yang, Xujun Zhang, Yu Kang, Hongyan Du, Zhenxing Wu, Jike Wang, Yundian Zeng, et al. Metalprognet: a structure-based deep graph model for metalloprotein–ligand interaction predictions. Chemical Science, 14(8):2054–2069, 2023.
- [Chen2021] Dong Chen, Jiaxin Zheng, Guo-Wei Wei, and Feng Pan. Extracting predictive representations from hundreds of millions of molecules. The Journal of Physical Chemistry Letters, 12(44):10793–10801, 2021.
- [Rives2021] Rives, Alexander, Joshua Meier, Tom Sercu, Siddharth Goyal, Zeming Lin, Jason Liu, Demi Guo et al. "Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences." Proceedings of the National Academy of Sciences 118, no. 15 (2021): e2016239118.
---
