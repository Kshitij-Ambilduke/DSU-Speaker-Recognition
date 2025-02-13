The DSU_creation folder contains the final datasets and all preprocessing scripts used for preparing the data.

To understand the structure of the Pandas training file, refer to gather_data/aligning_dsu2audio.ipynb.

``` bash
DSU_creation/  
├── LibriSpeech/  
│   ├── final_data_librispeech/   # Final LibriSpeech dataset  
├── VCTK/  
│   ├── final_data_vctk/          # Final VCTK dataset  
├── audio2dsu.sh                  # Script for converting audio to DSUs  
├── to_tsv.py                     # Script for converting data to TSV format  
├── to_wav.sh                     # Script for converting audio files to WAV format  
├── readme.md   

```