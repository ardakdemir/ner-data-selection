/usr/local/bin/nosh
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N generate_vectors

cd ~/ner-data-selection
singularity exec  --nv  --writable ~/singularity/pt-cuda-tf-tr-ft python ~/ner-data-selection/generate_vectors.py
