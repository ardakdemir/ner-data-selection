#!/bin/bash

save_folder="../dc_vectors_2401"
for dataset_name in "BC2GM" "BC5CDR" "JNLPBA" 'NCBI-disease' "BC4CHEMD" 'conll-eng'
do
  for aux_size in 1 2 3 5 10
  do
    my_save_folder=${save_folder}"_"${aux_size}"/"${dataset_name}
    exp_name="vectorgeneration_2401_"${dataset_name}
    echo ${my_save_folder}"  "${dataset_name}
    qsub -N $exp_name generate_vector_submit_2301.sh ${dataset_name} ${my_save_folder} ${aux_size}
  done

done