#!/bin/bash

save_folder="dc_vectors_2301/"
for dataset_name in "BC2GM" "BC5CDR" "JNLPBA" 'NCBI-disease' "BC4CHEMD" 'conll-eng'
do
  my_save_folder=${save_folder}${dataset_name}
  exp_name="vectorgeneration_2301_"${dataset_name}
  echo ${my_save_folder}"  "${dataset_name}
#  qsub -N $exp_name generate_vector_submit_2301.sh ${dataset_name} ${my_save_folder}
done