#!/bin/bash

save_root_prefix="/home/aakdemir/bioner_instancesim_0806_size_"
ner_save_prefix="/home/aakdemir/bioner_results_instancesim_0806_size_"
for size in 20000 30000 40000 50000 75000 100000
do
  dataset_save_folder_root=${save_root_prefix}${size}
  exp_name="bioner_instance_0806_size_"${size}
  ner_save_folder_root=${ner_save_prefix}${size}
  echo ${ner_save_folder_root}"  "${dataset_save_folder_root}
  qsub -N $exp_name trainner_submit.sh ${dataset_save_folder_root} ${size} ${ner_save_folder_root}
done