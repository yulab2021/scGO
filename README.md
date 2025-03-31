# scGO

[![python >3.8.13](https://img.shields.io/badge/python-3.8.13-brightgreen)](https://www.python.org/) 

### scGO, biologically informed deep learning for accurate cell status annotation
Accurate cell type annotation is crucial for downstream analysis of single-cell RNA sequencing data. However, existing annotation algorithms often face challenges such as inadequate handling of batch effects, absence of curated marker gene lists, and difficulty in utilizing latent gene-gene interaction information. Here, we present scGO, a machine learning technique for annotating cell status in single-cell RNA sequencing (scRNA-seq) data. Leveraging bioinformed neural networks, scGO exploits gene, transcription factor, and gene ontology associations, enhancing the interpretability of cell subtyping results. Through extensive experimentation on various scRNA-seq datasets, scGO demonstrates remarkable efficacy in precisely identifying and characterizing cell subtypes. Its versatility is showcased across diverse biological applications, including disease diagnosis, therapeutic target discovery, developmental stage prediction, and senescence status evaluation. This methodology not only unveils deeper insights into cellular diversity but also holds significant potential for addressing challenges in biology and clinical practice. For more information, including demo data and running examples, please refer to our online document [https://yulab2021.github.io/scGO_document/.](https://yulab2021.github.io/scGO_document/)

# Citing this work
If you use the code or data in this package, please cite:

You Wu, Pengfei Xu, Liyuan Wang, Shuai Liu, Yingnan Hou, Hui Lu, Peng Hu, Xiaofei Li, Xiang Yu, scGO: interpretable deep neural network for cell status annotation and disease diagnosis, Briefings in Bioinformatics, Volume 26, Issue 1, January 2025, bbaf018, https://doi.org/10.1093/bib/bbaf018


# Installation
The following modules are needed to run scGO.


module | version
---|---
python       |3.8.13
torch       |1.9.1
scanpy    |1.9.3
scikit-learn   |1.3.2
scipy    |1.10.1


Conda is recommended for package management, you can create a new conda environment and then install the packages. Here's an example of how you can do it. Create a new conda environment::
    
    conda create -n scGO python=3.8.13

Activate the newly created environment::

    conda activate scGO

Install the required modules::

    pip install torch==1.9.1
    pip install scanpy==1.9.3
    pip install scikit-learn==1.3.2
    pip install scipy==1.10.1


The entire installation will take about 1-5 minutes. After installing all the essential packages,  reset the environment's state by deactivating and reactivating the environment:
::
    conda deactivate
    conda activate scGO

We have also provided a yaml file in the repository so you can install the dependencies through the configuration file::

    conda env create -f scGO.yaml


The source code and data processing scripts can be downloaded by using the git clone command::

    git clone https://github.com/yulab2021/scGO.git


# Usage
### Training cell type annotation model

Train scGO model using [Baron dataset](https://www.cell.com/cell-systems/fulltext/S2405-4712(16)30266-6?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS2405471216302666%3Fshowall%3Dtrue). The Baron dataset covers a range of cell types found in the pancreas, including acinar cells, activated stellate cells, alpha cells, beta cells, delta cells, ductal cells, endothelial cells, epsilon cells, gamma cells, macrophages, mast cells, quiescent stellate cells, and schwann cells. Baron dataset is available at [GSE84133](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133). In this demo, a subsets of the the baron dataset was taken for demonstration purposes due to the large size of the original datasets. The demo dataset was located under ``./demo/`` directory.
``` 
    demo
    ├── baron_data.csv
    ├── baron_meta_data.csv
    ├── goa_human.gaf
    └── TF_annotation_hg38.demo.tsv
```

#### 1. Data normalization and filtering

Normalize data and filter out genes that expressed in fewer cells using script ``data_processing.py`` with command ``norm_and_filter``. The number of genes to be retained can be customized by using the argument ``--num_genes``. The default value is 2000. The normalized and filtered data will be saved in the ``./demo/`` folder.
```
    usage: data_processing.py norm_and_filter [-h] [--gene_expression_matrix GENE_EXPRESSION_MATRIX] [--num_genes NUM_GENES] [--output OUTPUT]
    optional arguments:
        -h, --help                                            Show this help message and exit
        --gene_expression_matrix GENE_EXPRESSION_MATRIX       Path to the input data
        --num_genes NUM_GENES                                 Number of top genes to keep
        --output OUTPUT                                       Path to the output data

    expample:
    python scripts/data_processing.py norm_and_filter --gene_expression_matrix demo/baron_data.csv --num_genes 2000 --output demo/baron_data_filtered.csv
```
##### 2. Building network connections

Build network connections between gene layer, TF layer and GO layer according to GO annotations and TF annotations using script ``data_processing.py`` with command ``build_network``. The human GO annotation used in this study is downloaded from the [Gene Ontology knowledgebase](https://doi.org/10.5281/zenodo.7504797). The connections between genes and TFs are builded according to the DAP-seq TF annotation data. The DAP-seq data is downloaded from the [Remap database](https://remap2022.univ-amu.fr/). The demo data offers a subset of the processed DAP-seq file for illustrative purposes.  The full processed DAP-seq file has been uploaded to [google drive](https://drive.google.com/file/d/1VPSDyNbs4lBITm2VoPD2eJ3BZGcdkrdC/view?usp=drive_link). 
```
    usage: data_processing.py build_network [-h] [--gene_expression_matrix GENE_EXPRESSION_MATRIX] [--GO_annotation GO_ANNOTATION] [--TF_annotation TF_ANNOTATION]
    optional arguments:
        -h, --help                                         Show this help message and exit
        --gene_expression_matrix GENE_EXPRESSION_MATRIX    Path to the input data
        --GO_annotation GO_ANNOTATION                      Path to the GO annotation file
        --TF_annotation TF_ANNOTATION                      Path to the TF annotation file
    
    example:
    python scripts/data_processing.py build_network --gene_expression_matrix demo/baron_data_filtered.csv --GO_annotation demo/goa_human.gaf  --TF_annotation demo/TF_annotation_hg38.demo.tsv
```

#### 3. Model training

To train the scGO model using your own dataset from scratch, you can run the ``scGO.py`` script with the command ``train``. scGO accepts both single-cell gene expression matrix (.csv) as input. Meta data (.csv) is needed for the training process. The column cell_type in the meta data is used as the label for training. You can specify the model save path by using the argument ``--model``. The model's training epochs can be defined using the argument ``--epochs``, and the model states will be saved at the end of each epoch. The training process duration can vary, depending on the size of your dataset and the computational capacity, and may range from minutes to several hours. Here is an example of the training process using the demo dataset.
```
    
    usage: scGO.py train [-h] --gene_expression_matrix GENE_EXPRESSION_MATRIX [--meta_data META_DATA] --model MODEL [--task TASK] [--epoch EPOCH] [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE]  [--label LABEL]

    optional arguments:
        -h, --help                                           Show this help message and exit  
        --gene_expression_matrix GENE_EXPRESSION_MATRIX      Train data file.
        --meta_data META_DATA                                Meta data, including cell type labels.
        --model MODEL                                        Output model file.
        --task TASK                                          Task type, classification or regression.
        --epoch EPOCH                                        Number of training epochs.
        --batch_size BATCH_SIZE                              Batch size.
        --learning_rate LEARNING_RATE                        Learning rate.
        --label LABEL                                        Column in meta_data to be predicted.

    expample:
    python scripts/scGO.py train --gene_expression_matrix demo/baron_data_filtered.csv --meta_data demo/baron_meta_data.csv --model models/scGO.demo.pkl
```

#### 4. Predicting new data

After the completion of the training process, the model file ``scGO.demo.pkl`` will be stored in the ``./models/`` folder. This trained model can be employed to make predictions on new data. Use the ``predict`` command to predict new data, and assign the predicted results using the ``--output`` argument.
```
    python scripts/scGO.py predict --gene_expression_matrix demo/baron_data.csv  --model models/scGO.demo.pkl --output demo/baron_data_filtered.predicted.csv
```
### Training regression model to predict a continuous cell status


Set the ``task`` argument to ``regression`` and specify the ``label`` argument to correspond to a column in the metadata that you aim to predict.
```
    python scripts/scGO.py train --gene_expression_matrix demo/baron_data_filtered.csv --task regression --epoch 100 --batch_size 8 --meta_data demo/baron_meta_data_senescence_score.csv --label senescence_score --model models/scGO.senescence_score.demo.pkl
```
To predict new data, load the pre-trained model, and generate predictions for the new data. The results should include both the predicted cell type label and the associated predicted value.
```
    python scripts/scGO.py predict --gene_expression_matrix demo/baron_data.csv --task regression --model models/scGO.senescence_score.demo.pkl --output demo/baron_meta_data_senescence_score.predicted.csv
```



# Time cost

The demo is expected to run for approximately 0-3 minutes. However, when applying scGO to larger datasets, the execution time may range from minutes to up to 30 minutes.

# Coypright

The tool is maintained by You Wu and Xiang Yu and is intended for non-commercial use. All rights reserved.

# Citation

To add.
