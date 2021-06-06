/usr/local/bin/nosh
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N dataselection_pipeline

cd ~/ner-data-selection
#singularity exec  --nv  --writable ~/singularity/pt-cuda-tf-tr-ft python ~/ner-data-selection/dataselection_pipeline.py
singularity exec  --nv  --writable ~/singularity/pt-cuda-tf-tr-ft python3 ~/ner-data-selection/generate_vectors.py
