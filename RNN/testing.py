import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
# from dataloader_dsu import LoadAudioDataset, MyCollate, get_loader
from dataloader_continuous import LoadAudioDataset, MyCollate, get_loader
from tqdm import tqdm
# from model_dsu import SpeakerRNN
# from model_dsu_lstm import SpeakerLSTM
from model import SpeakerRNN
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


batch_size = 32
PATH = "/mnt/data-artemis/kshitij/TEMP/Speaker-recognition/RNN/artifiacts/model/speaker_rnn_LS.pth"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SpeakerRNN(num_layers=3, bidirectional=True).to(DEVICE)
# model = SpeakerLSTM(num_layers=4, bidirectional=True, input_dim=128, hidden_dim=128).to(DEVICE) #lstm

model.load_state_dict(torch.load(PATH, weights_only=True, map_location=torch.device(DEVICE)))
print(model)
quit()
# dataLoader = get_loader(batch_size=batch_size, 
#                         df_path="/mnt/data-artemis/kshitij/TEMP/Speaker-recognition/DSU_creation/LibriSpeech/final_data_librispeech/train.csv")
# dataLoader = get_loader(batch_size=batch_size,
#                         audio_root="/mnt/data-poseidon/kshitij/gitlab/LS_audio/test_wav_files",
#                         df_path="/mnt/data-artemis/kshitij/TEMP/Speaker-recognition/DSU_creation/LibriSpeech/final_data_librispeech/test.csv")
dataLoader = get_loader(batch_size=batch_size, 
                              df_path="/mnt/data-artemis/kshitij/TEMP/Speaker-recognition/DSU_creation/LibriSpeech/final_data_librispeech/test.csv",
                              mapping_path="/mnt/data-artemis/kshitij/TEMP/Speaker-recognition/DSU_creation/LibriSpeech/speaker_label_mapping.json" )
    
def test(dataloader):
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for n, ((audios, lengths), labels) in tqdm(enumerate(dataloader), desc="iteration", total=len(dataloader)):
            audios = audios.to(DEVICE)
            labels = labels.to(DEVICE)
            
            output, labels = model(audios, lengths, labels)
            preds = torch.argmax(F.softmax(output, dim=1), dim=1)
            
            all_preds.extend(preds.cpu().numpy()) 
            all_labels.extend(labels.cpu().numpy()) 
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted') 
    recall = recall_score(all_labels, all_preds, average='weighted') 
    f1 = f1_score(all_labels, all_preds, average='weighted') 
    
    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

test(dataLoader)  


'''
Continuous speech results:    

LS: 
Accuracy: 0.9403
Precision: 0.9455
Recall: 0.9403
F1 Score: 0.9403

VCTK: 
Accuracy: 0.8932
Precision: 0.9021
Recall: 0.8932
F1 Score: 0.8939

Hyper-params:
model = SpeakerRNN(num_layers=3, bidirectional=True).to(DEVICE)
epochs = 25
LR = 0.0001
batch_size = 512
optimizer = AdamW  

DSU deduplication LSTM:

VCTK:
Accuracy: 0.3660
Precision: 0.3813
Recall: 0.3660
F1 Score: 0.3677  

LS:
Accuracy: 0.7358
Precision: 0.7522
Recall: 0.7358
F1 Score: 0.7348
'''



