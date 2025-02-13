import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from dataloader_dsu import LoadAudioDataset, MyCollate, get_loader
from tqdm import tqdm
import matplotlib.pyplot as plt

print("here")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("here")

class SpeakerRNN(nn.Module):
    def __init__(self, num_layers, input_dim=512, hidden_dim=128, num_classes=30, bidirectional=False):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=2001, embedding_dim=input_dim, padding_idx=2000)
        self.embed.weight.requires_grad = False  # Freeze the embedding layer

        # self.batch_norm = nn.BatchNorm1d(input_dim)
        self.rnn = nn.RNN(input_size=input_dim, 
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          nonlinearity='tanh',
                          batch_first=True,
                          bidirectional=bidirectional,
                          dropout=0.1)
        self.dropout = nn.Dropout(0.1)
        self.classifier_layer = nn.Sequential(
        nn.Linear(hidden_dim * (2 if bidirectional else 1), hidden_dim))
        
    def forward(self, audios, lengths, labels):
        lengths = torch.tensor(lengths)
        lengths, perm_idx = lengths.sort(0, descending=True)
        audios = audios[perm_idx]
        labels = labels[perm_idx]
        
        audios = self.embed(audios)
        # audios = self.batch_norm(audios.transpose(1, 2)).transpose(1, 2) 
        
        packed_input = pack_padded_sequence(audios, lengths, batch_first=True, enforce_sorted=False)
        packed_hidden_last_all_timesteps, all_hidden_layers_last_timestep = self.rnn(packed_input) 
        
        (all_hidden, lengths) = pad_packed_sequence(packed_hidden_last_all_timesteps, batch_first=True)
        
        batch_indices = torch.arange(len(lengths))  
        last_time_steps = lengths - 1
        last_hidden_states = all_hidden[batch_indices, last_time_steps]
        
        last_hidden_states = self.dropout(last_hidden_states)
        output = self.classifier_layer(last_hidden_states)
        return output, labels

def train(model, train_dataloader, val_dataloader, criterion, epochs, optimizer, batch_size=2, clip=1.0):
    epch_loss_list = []
    best_val_acc = 0 
    best_model = None  
    val_loss_list = []
    
    for epc in tqdm(range(epochs), desc="epochs", total=epochs):  
        model.train() 
        epch_loss = 0      
        
        for n, ((audios, lengths), labels) in tqdm(enumerate(train_dataloader), desc="iteration", total=len(train_dataloader)):
            audios = audios.to(DEVICE)
            labels = labels.to(DEVICE)
            output, labels = model(audios, lengths, labels)
            loss = criterion(output, labels.long())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epch_loss += loss.item()

        avg_train_loss = epch_loss / (n+1)
        print(f"Epoch {epc+1} Train Loss: {avg_train_loss:.4f}")
        epch_loss_list.append(avg_train_loss)
        
        val_loss, val_acc = validate(model, val_dataloader, criterion)
        print(f"Epoch {epc+1} Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        val_loss_list.append(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
    
    if best_model:
        torch.save(best_model, "/mnt/data-artemis/kshitij/TEMP/speaker_rnn_best_notrainembed_vctk.pth")
    
    return epch_loss_list, val_loss_list


def validate(model, val_dataloader, criterion):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for ((audios, lengths), labels) in tqdm(val_dataloader, desc="Validation", total=len(val_dataloader)):
            audios = audios.to(DEVICE)
            labels = labels.to(DEVICE)
            output, _ = model(audios, lengths, labels)
            
            loss = criterion(output, labels.long())
            total_loss += loss.item()
            
            _, preds = torch.max(output, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(val_dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()
    print("Making model...")
    
    model = SpeakerRNN(num_layers=4, bidirectional=True, input_dim=512, hidden_dim=256).to(DEVICE)
    print(model)
    epochs = 70
    LR = 0.001
    batch_size = 512
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    
    train_loader = get_loader(batch_size=batch_size, df_path="/mnt/data-artemis/kshitij/TEMP/Speaker-recognition/DSU_creation/VCTK/final_data_vctk/train.csv")
    val_loader = get_loader(batch_size=1, df_path="/mnt/data-artemis/kshitij/TEMP/Speaker-recognition/DSU_creation/VCTK/final_data_vctk/dev.csv")  
    
    print("Starting training...")
    
    epoch_loss_list, val_loss_list = train(model,
                                            train_loader,
                                            val_loader,
                                            criterion,
                                            epochs,
                                            optimizer,
                                            batch_size)
    
    torch.save(model.state_dict(), "/mnt/data-artemis/kshitij/TEMP/speaker_rnn_vctk_DSU_Dedup.pth")
    
    save_path = "/mnt/data-artemis/kshitij/TEMP/speaker_rnn_best_notrainembed_vctk.png"

    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), epoch_loss_list, label='Train Loss', color='blue')
    plt.plot(range(epochs), val_loss_list, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

    print(f"Plot saved at: {save_path}")