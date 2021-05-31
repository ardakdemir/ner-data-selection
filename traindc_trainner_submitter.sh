#!/bin/bash

dataset_root="../biobert_data/datasets/BioNER_2804_labeled_combined"
#Train models
for i in 1 2 3 5 10
do
  dc_save_folder="../dc_result_3105_"${i}
  ner_save_folder="../ner_result_3105_"${i}
  dc_dataset_root="../biobert_data/datasets/BioNER_2505_DC_datasets_relsize_"${i}
  ner_dataset_root="../instanceselection_2005_labeled/BioBERT"
  exp_name="BioNER_2505_DC_datasets_relsize_"${i}
  qsub -N $exp_name traindc_trainner_submit.sh ${dc_save_folder} ${ner_save_folder} ${dc_dataset_root} ${ner_dataset_root}
done
