/usr/local/bin/nosh
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
cd ~/ner-data-selection

save_folder="../alldata_2202_nerresults"
dataset_root="../biobert_data/datasets/BioNER_2804_labeled_combined/BC2GM"


#Train models
singularity exec  --nv  --writable ~/singularity/pt-cuda-tf-tr-ft python ~/ner-data-selection/train_nermodels.py --save_folder_root ${save_folder} --save_folder ${save_folder} --dataset_root ${dataset_root}  --evaluate_root ${dataset_root}
