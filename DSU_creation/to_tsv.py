import os
import numpy as np
import math
from scipy.io import wavfile
from tqdm import tqdm
import argparse

def process_audio_files(audio_files_dir, output_root, file_name):

    os.makedirs(output_root, exist_ok=True)
    audio_list = os.listdir(audio_files_dir)
    file_name = os.path.join(output_root, f"{file_name}.tsv")
    faulty = 0
    
    with open(file_name, 'w+') as f:
        f.write(audio_files_dir+'/\n')
        
        for audio_file in audio_list:
            try:
                _, nparr = wavfile.read(os.path.join(audio_files_dir, audio_file))
                num_samples = str(len(nparr))

                towrite = audio_file + '\t' + num_samples + '\n'
                f.write(towrite)
            except Exception as e:
                print(f"Error processing file {audio_file}: {e}")
                faulty += 1

    print("Total faulty audio files:", faulty)

def main():
    parser = argparse.ArgumentParser(description="Process audio files into splits and save their sample counts in TSV files.")
    
    parser.add_argument("--audio_files_dir", type=str, required=True, help="Path to the directory containing audio files.")
    parser.add_argument("--output_root", type=str, required=True, help="Path to the directory where output TSV files will be saved.")
    parser.add_argument("--file_name", type=str, required=True, help="name of file")
    
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    process_audio_files(args.audio_files_dir, args.output_root, args.file_name)

if __name__ == "__main__":
    main()