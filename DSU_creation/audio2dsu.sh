#!/bin/bash

source /mnt/data-artemis/kshitij/TEMP/temp_env/bin/activate         #activate your environment (you might have to do conda activate speaker_ver if you installed conda env)
cd /mnt/data-artemis/kshitij/TEMP/fairseq/examples/hubert/simple_kmeans       #Change to fairseq path

ckpt_path=/mnt/scratch-artemis/kshitij/clustering/feature_extraction/model/hubert_large_ll60k.pt

layer=22
nshard=1
rank=0
n_cluster=2000          #Numbber of clusters
feat_dir=/mnt/data-poseidon/kshitij/Speaker-recognition/feat_dir            #Directory to extract features to

# km_path=DSU_creation/LibriSpeech/kmeans_model_L22_C2000/speaker_recog_kmeans_L22.model
km_path=DSU_creation/VCTK/kmeans_model/speaker_recog_kmeans_L${layer}_C${n_cluster}.model        #kmeans model path (to be created)

# lab_dir=DSU_creation/LibriSpeech/kmeans_model_L22_C2000_output          #Output kmeans directory
lab_dir=DSU_creation/VCTK/kmeans_dir_L${layer}_C${n_cluster}_output          #Output kmeans directory

# tsv_dir=DSU_creation/LibriSpeech/tsv_dir         #TSV file of audios to convert to kmeans  
tsv_dir=DSU_creation/VCTK/tsv_dir         #TSV file of audios to convert to kmeans  

split=audio_dev        #tsv file name without extension

mkdir $feat_dir
mkdir $lab_dir

# Extract HuBERT features
python dump_hubert_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}

## Learning k-means
# python learn_kmeans.py ${feat_dir} ${split} ${nshard} ${km_path} ${n_cluster} --percent -1 --batch_size 12000

echo "Saved kmeans at ${km_path}"
# Applying k-means
python dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}

echo "Process done!"