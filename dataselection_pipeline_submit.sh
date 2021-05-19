/usr/local/bin/nosh
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G

mode=${1}
cd ~/ner-data-selection

echo "Training for "${mode}
singularity exec  --nv  --writable ~/singularity/pt-cuda-tf-tr-ft python ~/ner-data-selection/dataselection_pipeline.py --selection_method $mode
