import json
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from itertools import groupby
from torch.nn.utils.rnn import pad_sequence
import os


SPEAKER_MAPPING_PATH = "/mnt/data-artemis/kshitij/TEMP/Speaker-recognition/DSU_creation/VCTK/speaker_label_mapping.json"  


class LoadAudioDataset(Dataset):
    def __init__(self, df_path, mapping_path=SPEAKER_MAPPING_PATH, layer=22):
        '''
        df_path: Path to CSV file for dataset
        mapping_path: JSON file containing speaker ID mappings
        '''
        audio_df = pd.read_csv(df_path)

        if os.path.exists(mapping_path):
            with open(mapping_path, "r") as f:
                speaker_mapping = json.load(f)
            print("Loaded existing speaker mapping.")
        else:
            speaker_mapping = {str(spk): idx for idx, spk in enumerate(sorted(audio_df['speaker_id'].unique()))}
            with open(mapping_path, "w") as f:
                json.dump(speaker_mapping, f)
            print("Created and saved new speaker mapping.")
            
        audio_df["speaker_id"] = audio_df["speaker_id"].astype(str)
        audio_df['label'] = audio_df['speaker_id'].map(speaker_mapping)
        # print("asasas")
        # print(audio_df['label'])
        self.audio_df = self.deduplicate_dsu(audio_df)

    def __len__(self):
        return self.audio_df.shape[0]

    def __getitem__(self, index):
        dsu = self.audio_df.iloc[index]['DSU_Deduplicated']
        label = self.audio_df.iloc[index]['label']
        dsu = [int(i) for i in dsu.split()]
        return {"audio": torch.tensor(dsu), "label": label}

    def deduplicate_dsu(self, df):
        df['DSU_Deduplicated'] = df['DSU'].apply(lambda x: ' '.join([k for k, _ in groupby(x.split())]))
        return df


class MyCollate:
    
    def __init__(self):
        pass
    
    def __call__(self, batch):
        audios = [i['audio'] for i in batch]
        # print(audios)
        lengths = [len(audios[i]) for i in range(len(audios))]
        labels = torch.tensor([i['label'] for i in batch])
        audios = pad_sequence(audios, batch_first=True, padding_value=2000)
        return (audios, lengths), labels
        
        
def correct_audio_path(x, root):
    x = os.path.basename(x)
    return os.path.join(root, x)


def get_loader(
    df_path = "/mnt/data-artemis/kshitij/TEMP/Speaker-recognition/DSU_creation/LibriSpeech/final_data_librispeech/train.csv",
    mapping_path="/mnt/data-artemis/kshitij/TEMP/Speaker-recognition/DSU_creation/LibriSpeech/speaker_label_mapping.json",
    batch_size=2,
    num_workers=0,
    shuffle=True,
    pin_memory=True
):

    dataset = LoadAudioDataset(
        df_path=df_path,
        mapping_path=mapping_path
        )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate())

    return loader

#testing
if __name__=="__main__":
    dataLoader = get_loader()
    for (audios, lengths), labels in dataLoader:
        print(audios.shape)
        print(audios[0])
        print(audios[1])
        print(labels)
        print(lengths)
        break
