/usr/local/bin/nosh
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
cd ~/ner-data-selection

dc_save_folder=${1}
ner_save_folder=${2}
dc_dataset_root=${3}
ner_dataset_root=${4}

echo $dc_save_folder"  "$ner_save_folder"  "$dc_dataset_root"  "$ner_dataset_root
singularity exec  --nv  --writable ~/singularity/pt-cuda-tf-tr-ft python ~/ner-data-selection/train_dc_models.py --save_folder_root ${dc_save_folder} --save_folder ${dc_save_folder} --dataset_root ${dc_dataset_root}  --evaluate_root ${dc_dataset_root} --multiple
singularity exec  --nv  --writable ~/singularity/pt-cuda-tf-tr-ft python ~/ner-data-selection/train_nermodels.py --dc_save_root ${dc_save_folder} --save_folder_root ${ner_save_folder} --save_folder ${ner_save_folder} --dataset_root ${ner_dataset_root}  --evaluate_root ${ner_dataset_root} --load_dc_model  --multiple
