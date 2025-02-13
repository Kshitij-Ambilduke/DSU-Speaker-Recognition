import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from dataloader_continuous import LoadAudioDataset, MyCollate, get_loader
from tqdm import tqdm
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SpeakerRNN(nn.Module):
    def __init__(self, num_layers, input_dim=1024, hidden_dim=256, num_classes=30, bidirectional=False):
        super().__init__()
        
        self.rnn = nn.RNN(input_size=input_dim, 
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          nonlinearity='tanh',
                          batch_first=True,
                          bidirectional=bidirectional)
        self.classifier_layer = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)
    
    def forward(self, audios, lengths, labels):
        lengths = torch.tensor(lengths)
        lengths, perm_idx = lengths.sort(0, descending=True)
        audios = audios[perm_idx]
        labels = labels[perm_idx]
        
        packed_input = pack_padded_sequence(audios, lengths, batch_first=True, enforce_sorted=False)
        packed_hidden_last_all_timesteps, _ = self.rnn(packed_input)
        
        (all_hidden, lengths) = pad_packed_sequence(packed_hidden_last_all_timesteps, batch_first=True)
        
        batch_indices = torch.arange(len(lengths))
        last_time_steps = lengths - 1
        last_hidden_states = all_hidden[batch_indices, last_time_steps]
        
        output = self.classifier_layer(last_hidden_states)
        return output, labels


def evaluate(model, dataloader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for (audios, lengths), labels in dataloader:
            audios, labels = audios.to(DEVICE), labels.to(DEVICE)
            output, labels = model(audios, lengths, labels)
            loss = criterion(output, labels.long())
            val_loss += loss.item()
            preds = torch.argmax(F.softmax(output, dim=1), dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = val_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def train(model, train_loader, val_loader, criterion, epochs, optimizer, batch_size=2, clip=2.0):
    best_val_loss = float('inf')
    train_loss_list, val_loss_list = [], []
    
    for epc in tqdm(range(epochs), desc="Epochs", total=epochs):  
        model.train()
        epch_loss = 0      
        for (audios, lengths), labels in tqdm(train_loader, desc="Iteration", total=len(train_loader)):
            audios, labels = audios.to(DEVICE), labels.to(DEVICE)
            output, labels = model(audios, lengths, labels)
            loss = criterion(output, labels.long())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epch_loss += loss.item()
        
        train_loss = epch_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        
        print(f"Epoch {epc+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "/mnt/data-artemis/kshitij/TEMP/zsaved_things/model/speaker_rnn_best_cont_LS.pth")
            print("Best model saved!")
    
    return train_loss_list, val_loss_list


if __name__ == '__main__':
    
    criterion = nn.CrossEntropyLoss()
    print("Making model...")
    
    model = SpeakerRNN(num_layers=3, bidirectional=True).to(DEVICE)
    epochs = 30
    LR = 0.0001
    batch_size = 1024
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)        
    
    train_loader = get_loader(batch_size=batch_size, audio_root="/mnt/data-poseidon/kshitij/gitlab/LS_audio/train_wav_files",
    df_path="/mnt/data-artemis/kshitij/TEMP/Speaker-recognition/DSU_creation/LibriSpeech/final_data_librispeech/train.csv",shuffle=True)
    
    val_loader = get_loader(batch_size=batch_size, audio_root="/mnt/data-poseidon/kshitij/gitlab/LS_audio/dev_wav_files",
    df_path="/mnt/data-artemis/kshitij/TEMP/Speaker-recognition/DSU_creation/LibriSpeech/final_data_librispeech/dev.csv", shuffle=True)
    
    print("Starting training...")
    
    train_loss_list, val_loss_list = train(model, train_loader, val_loader, criterion, epochs, optimizer, batch_size)
    
    torch.save(model.state_dict(), "/mnt/data-artemis/kshitij/TEMP/zsaved_things/model/speaker_rnn_final_cont_LS.pth")
    
    save_path = "/mnt/data-artemis/kshitij/TEMP/zsaved_things/plots/training_cont_LS.png"
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), train_loss_list, label='Train Loss', color='blue')
    plt.plot(range(epochs), val_loss_list, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

    print(f"Plot saved at: {save_path}")