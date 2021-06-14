#!/bin/bash

#dataset_root="../biobert_data/datasets/BioNER_2804_labeled_combined"
#Train models

ner_save_folder="/home/aakdemir/dataselect_lda_nerresult_1406"

for dataset_name in 's800' 'NCBI-disease' 'JNLPBA' 'linnaeus' 'BC4CHEMD' 'BC2GM' 'BC5CDR' 'conll-eng'
do
  exp_name="BioNER_lda_1306_"${dataset_name}
  dataset_root="/home/aakdemir/lda_selecteddata_1306/LDA/"${dataset_name}
  echo $dataset_name"  "${dataset_root}
  qsub -N $exp_name trainner_submit.sh  ${dataset_root} 0 ${ner_save_folder}
done
