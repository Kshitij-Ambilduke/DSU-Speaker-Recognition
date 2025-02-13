# Speaker recoginition using Discrete speech units
Prepared by: 
Kshitij AMBILDUKE, Elizaveta SIROTINA, Polina SOLOVEVA

M1 AI, Université Paris-Saclay, Orsay

The datasets we used in this project are:
1. LibriSpeech [(Link)](https://huggingface.co/datasets/openslr/librispeech_asr)
2. VCTK [(Link)](https://datashare.ed.ac.uk/handle/10283/2651)

## Repository structure
Here is the structure of the repository highlighting the important folders and files. Each folder contains its own readme file with more detailed specifications
``` bash
Speaker-recognition/
.
├── DSU_creation
│   ├── LibriSpeech
│   │   └── final_data_librispeech/     #Contains the LibriSpeech final dataset
│   ├── VCTK
│   │   └── final_data_vctk/            #Contains the VCTK final dataset
│   ├── audio2dsu.sh
│   ├── readme.md
│   ├── to_tsv.py
│   └── to_wav.sh
├── RNN
│   ├── artifiacts
│   │   ├── model
│   │   └── plots
│   ├── dataloader_continuous.py
│   ├── dataloader_dsu.py
│   ├── model.py
│   ├── model_dsu.py
│   ├── model_dsu_lstm.py
│   ├── readme.md
│   ├── temp.ipynb
│   └── testing.py
├── Vectorizers
│   ├── CV.ipynb
│   ├── CV_vanilla_bigrams.csv
│   ├── CV_vanilla_unigrams.csv
│   ├── Tfidf.ipynb
│   └── vectorizer_module.py
├── gather_data
│   ├── LibriSpeechData.ipynb
│   ├── VCTKData.ipynb
│   ├── aligning_dsu2audio.ipynb
├── data_exploration.ipynb            #Notebook with all main data exploration
├── readme.md
└── requirements.txt
```

## Installation

Pre-requisites:
``` bash
conda create -n speaker_ver python=3.10.16
conda activate speaker_ver
cd Speaker-recog
pip install -r requirements.txt
```

To view the train dataset run (Recommended):
``` python
import pandas as pd
train_data_ls = pd.read_pickle("DSU_creation/LibriSpeech/final_data_librispeech/train.csv")
train_data_vctk = pd.read_pickle("DSU_creation/VCTK/final_data_librispeech/train.csv")
```

The exact audio files used in the experiments can be found here: [Gitlab](https://gitlab.com/Kshitij-Ambilduke/speaker-recognition)

## Converting audio files into DSUs:
```bash
git clone https://github.com/pytorch/fairseq
pip install fairseq
cd fairseq
pip install --editable ./
```

Reference for creating DSUs: [Fairseq kmeans documentation](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert/simple_kmeans)

> Make sure all the audio files are in `.wav` format and sampled at 16kHz (same as the sampling rate at which the audio encoder is trained on). `DSU_creation/to_wav.sh` can be used to convert audio files in a folder to the desired format (`.wav` and 16kHz sampling).

Follow these steps for getting the DSUs:
1. Ready the tsv file using `DSU_creation/to_tsv.py`
2. Download the model used for feature extraction (We used HuBERT Large [download_link](https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k_finetune_ls960.pt))
3. Then use the `DSU_creation/audio2dsu.sh` file to first extract the features from a specific layer from the audio model and then apply the already trained kmeans-model (for LibriSpeech: `DSU_creation/LibriSpeech/kmeans_model_L22_C2000/speaker_recog_kmeans_L22.model` and for VCTK: `DSU_creation/VCTK/kmeans_model/speaker_recog_kmeans_L22_C2000.model`) to these extracted features to obtain the DSUs.

Following are the arguments for the `DSU_creation/audio2dsu.sh` file:
```
ckpt_path = Path to the HuBERT model
layer = Layer to extract features from
feat_dir = Path to the directory where the features will be dumped
km_path = Path to the kmeans model        
lab_dir = Output directory for saving the file with Discrete speech units
tsv_dir = Path to the TSV directory created using "DSU_creation/to_tsv.py"
split = Name of the TSV file without the extension.
```
Besides these the arguments `nshard, rank, n_cluster` should be left as it is as they are concerned with either training the kmeans model or for multiprocessing of DSUs.

At the end of this, you will have:
1. A TSV file with audio file locations .
2. The corresponding, aligned Discrete Speech Units. 