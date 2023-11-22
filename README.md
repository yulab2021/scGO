# scGO

[![python >3.8.13](https://img.shields.io/badge/python-3.8.13-brightgreen)](https://www.python.org/) 

### scGO, biologically informed deep learning for accurate cell status annotation
Accurate cell type annotation is crucial for downstream analysis of single-cell RNA sequencing data. However, existing annotation algorithms often face challenges such as inadequate handling of batch effects, absence of curated marker gene lists, and difficulty in utilizing latent gene-gene interaction information. Here, we present scGO, a machine learning technique for annotating cell status in single-cell RNA sequencing (scRNA-seq) data. Leveraging bioinformed neural networks, scGO exploits gene, transcription factor, and gene ontology associations, enhancing the interpretability of cell subtyping results. Through extensive experimentation on various scRNA-seq datasets, scGO demonstrates remarkable efficacy in precisely identifying and characterizing cell subtypes. Its versatility is showcased across diverse biological applications, including disease diagnosis, therapeutic target discovery, developmental stage prediction, and senescence status evaluation. This methodology not only unveils deeper insights into cellular diversity but also holds significant potential for addressing challenges in biology and clinical practice. For more information, including demo data and running examples, please refer to our online document [https://www.biorxiv.org/content/10.1101/2021.12.05.471261v1.](https://www.biorxiv.org/content/10.1101/2021.12.05.471261v1)


# Installation
The following modules are needed to run scGO.


.. list-table:: Required modules
   :widths: 50 50
   :header-rows: 1

   * - module
     - version
   * - python 
     - 3.8.13
   * - torch
     - 1.9.1


Conda is recommended for package management, you can create a new conda environment and then install the packages. Here's an example of how you can do it. Create a new conda environment::
    
    conda create -n scGO python=3.8.13

Activate the newly created environment::

    conda activate scGO

Install the required modules::

    pip install torch==1.9.1
    pip install scanpy==1.9.3
    



The entire installation will take about 1-5 minutes. After installing all the essential packages,  reset the environment's state by deactivating and reactivating the environment:
::
    conda deactivate
    conda activate scGO

We have also provided a yaml file in the repository so you can install the dependencies through the configuration file::

    conda env create -f scGO.yaml


The source code and data processing scripts can be downloaded by using the git clone command::

    git clone https://github.com/yulab2021/scGO.git


# Usage





# Time cost

# Coypright

The tool is maintained by You Wu and Xiang Yu and is intended for non-commercial use. All rights reserved.

# Citation

[to do]