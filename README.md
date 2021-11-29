# UDON: Unsupervised Data selectiON for Biomedical Entity Recognition


This repository contains the code and the data accompanying our ICCBD 2021 submission, 
***UDON: Unsupervised Data selectiON for Biomedical Entity Recognition***.  


## Setup

We provide a docker image to relieve the need for installing python dependencies for running the code.

Pull image:  
```
docker pull aakdemir/pytorch-cuda-tensorflow
```

Run a docker shell:  

```
docker run -v <LOCAL_TO_MOUNT>:<PATH_INSIDE_CONTAINER> -it aakdemir/pytorch-cuda-tensorflow:latest /bin/bash
```



## Data

All datasets used in our experiments are available [here](https://drive.google.com/file/d/1iZ3Jv1RrChbbxB0vaQHIrPw5EftjHzds/view?usp=sharing) (11Mb).
The dataset contains 8 datasets:

- s800
- NCBI-disease
- JNLPBA
- linnaeus
- BC4CHEMD
- BC2GM
- BC5CDR
- conll-eng



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


***Domain Classification for all Models.*** 

Please update the ```path``` variable to point to the ```allsentences_pickle.p``` pickle file that contains the sentences encodings for all 4 LMs.  
You can generate these encodings by running the ```generate_vectors.py ``` script for all models.

```
python3 domain_classification.py
```