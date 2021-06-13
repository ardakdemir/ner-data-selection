/usr/local/bin/nosh
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
cd ~/ner-data-selection

save_folder_root=${1}
size=${2}
ner_save_folder=${3}


#singularity exec --nv  --writable ~/singularity/pt-cuda-tf-tr-ft python ~/ner-data-selection/generate_subsets.py --save_folder ${save_folder_root} --size ${size}
singularity exec --nv  --writable ~/singularity/pt-cuda-tf-tr-ft python ~/ner-data-selection/train_nermodels.py  --save_folder_root ${ner_save_folder} --save_folder ${ner_save_folder} --dataset_root ${save_folder_root} --evaluate_root ${save_folder_root}