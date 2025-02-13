from transformers import Wav2Vec2Processor, HubertModel
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import os
import torchaudio

cache_dir = "/mnt/data-artemis/kshitij/TEMP/ls-cache"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(DEVICE)

PROCESSOR = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft", cache_dir=cache_dir)


MODEL = HubertModel.from_pretrained("facebook/hubert-large-ll60k",
                                            cache_dir=cache_dir).to(DEVICE)

class LoadAudioDataset(Dataset):
    '''
    audio_root: {
        "/mnt/data-poseidon/kshitij/gitlab/VCTK_audio/train_wav_16khz",
        "/mnt/data-poseidon/kshitij/gitlab/LS_audio/train_wav_files"
    }
    df_path: {
        "/mnt/data-artemis/kshitij/TEMP/Speaker-recognition/DSU_creation/VCTK/final_data_vctk/train.csv",
        "/mnt/data-artemis/kshitij/TEMP/Speaker-recognition/DSU_creation/LibriSpeech/final_data_librispeech/train.csv",
    }
    '''
    def __init__(self, audio_root, df_path, layer=22):
        '''
        audio_root: root of the audio folder where all the audio files are stored
        df_path: path to the CSV file for the dataset
        '''
        
        self.audio_root = audio_root
        self.layer = layer
        
        df = pd.read_csv(df_path)
        df['file'] = df['file'].apply(lambda x: correct_audio_path(x, self.audio_root))
        df['label'] = pd.Categorical(df['speaker_id']).codes
        
        self.audio_df = df
    
    def __len__(self):
        return self.audio_df.shape[0]
    
    def __getitem__(self, index):
        audio_location = self.audio_df.iloc[index]['file']
        label = self.audio_df.iloc[index]['label']
        waveform, sample_rate = torchaudio.load(audio_location)  
        
        if sample_rate != 16000:
            transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000).to(DEVICE)
            waveform = transform(waveform)
            
        waveform = waveform.to(DEVICE)
        with torch.no_grad():
            hidden_states = MODEL(waveform, output_hidden_states=True).hidden_states[self.layer].to('cpu')
        return {"audio":hidden_states, "label":label}


class MyCollate:
    
    def __init__(self):
        pass
    
    def __call__(self, batch):
        audios = [i['audio'].squeeze() for i in batch]
        lengths = [audios[i].shape[0] for i in range(len(audios))]
        labels = torch.tensor([i['label'] for i in batch])
        audios = pad_sequence(audios, batch_first=True, padding_value=0)
        return (audios, lengths), labels
        
        
def correct_audio_path(x, root):
    x = os.path.basename(x)
    return os.path.join(root, x)


def get_loader(
    audio_root="/mnt/data-poseidon/kshitij/gitlab/VCTK_audio/train_wav_16khz",
    df_path="/mnt/data-artemis/kshitij/TEMP/Speaker-recognition/DSU_creation/VCTK/final_data_vctk/train.csv",
    batch_size=2,
    num_workers=0,
    shuffle=True,
    pin_memory=True
):

    dataset = LoadAudioDataset(audio_root,
                            df_path)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate())

    return loader

# if __name__=="__main__":
    # dataLoader = get_loader()
    # for (audios, lengths), labels in dataLoader:
        # print(audios.shape)
        # print(audios[0])
        # print(audios[1])
        # print(labels)
        # break
