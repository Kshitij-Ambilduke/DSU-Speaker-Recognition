#!/bin/bash

input_dir="/mnt/data-poseidon/kshitij/Speaker-recognition/train_wav"  
output_dir="/mnt/data-poseidon/kshitij/Speaker-recognition/train_wav_16khz" 

mkdir -p "$output_dir"

for file in "$input_dir"/*.{mp3,flac,ogg,opus,m4a,wma,wav}; do
    if [ -f "$file" ]; then
        filename=$(basename -- "$file")
        filename_noext="${filename%.*}"

        ffmpeg -i "$file" -ar 16000 "$output_dir/${filename_noext}.wav" -loglevel error
        
        echo "Converted: $file -- $output_dir/${filename_noext}.wav"
    fi
done

echo "All files Train successfully!"

input_dir="/mnt/data-poseidon/kshitij/Speaker-recognition/dev_wav"  
output_dir="/mnt/data-poseidon/kshitij/Speaker-recognition/dev_wav_16khz" 

mkdir -p "$output_dir"

for file in "$input_dir"/*.{mp3,flac,ogg,opus,m4a,wma,wav}; do
    if [ -f "$file" ]; then
        filename=$(basename -- "$file")
        filename_noext="${filename%.*}"

        ffmpeg -i "$file" -ar 16000 "$output_dir/${filename_noext}.wav" -loglevel error
        
        echo "Converted: $file -- $output_dir/${filename_noext}.wav"
    fi
done

echo "All files Dev successfully!"


#!/bin/bash

input_dir="/mnt/data-poseidon/kshitij/Speaker-recognition/test_wav"  
output_dir="/mnt/data-poseidon/kshitij/Speaker-recognition/test_wav_16khz" 

mkdir -p "$output_dir"

for file in "$input_dir"/*.{mp3,flac,ogg,opus,m4a,wma,wav}; do
    if [ -f "$file" ]; then
        filename=$(basename -- "$file")
        filename_noext="${filename%.*}"

        ffmpeg -i "$file" -ar 16000 "$output_dir/${filename_noext}.wav" -loglevel error
        
        echo "Converted: $file -- $output_dir/${filename_noext}.wav"
    fi
done

echo "All files Test successfully!"
