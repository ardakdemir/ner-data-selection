# UDON: Unsupervised Data selectiON for Biomedical Entity Recognition


This repository contains the code and the data accompanying our CoNLL 2021 submission, ***UDON: Unsupervised Data selectiON for Biomedical Entity Recognition***.  
For the blind-review phase, we redact all private/personal information.


## Data

All datasets used in our experiments is available [here](https://drive.google.com/file/d/1iZ3Jv1RrChbbxB0vaQHIrPw5EftjHzds/view?usp=sharing) (11Mb).
The dataset contains 8 datasets:

- s800
- NCBI-disease
- JNLPBA
- linnaeus
- BC4CHEMD
- BC2GM
- BC5CDR
- conll-eng


## Code

All the source code to replicates is available [here](https://drive.google.com/file/d/1OD-72i7G0tVbbp43DcaDlRh1F4puZOTB/view?usp=sharing).
- ***source_code*** folder contains all the codes for running the experiments.  
- ***helper_notebooks*** folder contains two jupyter notebooks for generating the visualizations and tables in the paper.

### How to run the code.


***Selecting data.***

```

python3 generate_vectors.py --root_folder <Folder containing the all the datasets> --selected_save_root <Folder to store the selected sentences> 

```



***Train NER Model.*** 

For all datasets and models (BioWordVec, RoBERTa, distilBERT, BioBERT)
```
python3 train_nermodels.py  --dataset_root <Root folder containing all training data> --evaluate_root <Root folder containing all test data>  --multiple --multi_model

```

***Inference mode for NER.*** 

```
python3 train_nermodels.py  --inference  --dataset_root <PATH_TO_TEST_FOLDER> --evaluate_root <PATH_TO_TEST_FOLDER> --model_path <<PATH_TO_SAVED_MODEL> --class_dict_path <PATH_TO_CLASS_TO_IDX_FILE>

```

***Pretrain BioBERT for Domain Classification.*** 


1- Generate DC datasets. Update the ```ROOT_FOLDER``` and ```SAVE_FOLDER``` values inside the script accordingly. 


```
python3 generate_dc_datasets.py

```

2- Train.  
```
python3 train_dc_models.py  --multiple  --save_folder <PATH_TO_STORE_THE_TRAINED_MODELS> --dataset_root <Root folder containing all training data> --evaluate_root <Root folder containing all test data>

```