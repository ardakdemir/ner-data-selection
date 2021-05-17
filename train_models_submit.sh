/usr/local/bin/nosh
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
cd ~/ner-data-selection

save_folder="../selection_labeled_1005_randoms"
dataset_root="../dataselection_1005_labeled"
singularity exec  --nv  --writable ~/singularity/pt-cuda-tf-tr-ft python ~/ner-data-selection/train_nermodels.py --multiple --multi_model --save_folder_root ${save_folder} --dataset_root ${dataset_root}  --evaluate_root ${dataset_root}
