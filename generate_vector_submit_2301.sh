/usr/local/bin/nosh
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N generate_vectors
dataset_name=${1}
save_folder=${2}
aux_size=${3}
size=3000
cd ~/ner-data-selection
singularity exec  --nv  --writable ~/singularity/pt-cuda-tf-tr-ft python3 generate_vectors.py --model_load_path ../dc_result_3105_${aux_size}/${dataset_name}/best_model_weights.pkh  --save_folder ${save_folder}

