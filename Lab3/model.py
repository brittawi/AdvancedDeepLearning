import torch
import torch.nn as nn
import torchvision.models as models
import lightning as L

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False) -> None:
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, images):
        features = self.inception(images)
        return self.dropout(self.relu(features[0]))
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers) -> None:
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

class CNN2RNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNN2RNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0) # Sets image features from CNN as input to RNN
            states = None

            # Predicts each word of the caption
            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(0)
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0) # Takes predicted word and sets as input for next time step

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]

class ImageCaptioner(L.LightningModule):
    def __init__(self, ic_model, dataset, criterion):
        super().__init__()
        self.CNNtoRNN = ic_model
        self.dataset = dataset
        self.criterion = criterion

        #Only train fc layer
        for name, param in self.CNNtoRNN.encoderCNN.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def training_step(self, batch, batch_idx):
        imgs, captions = batch
        outputs = self.CNNtoRNN(imgs, captions[:-1]) # Don't send in last one because that what we want to predict
        loss = self.criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)) # reshapes from (seq_len, N, voc_size) to seq_len and N being concatenated)
        self.log("Traning loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self, learning_rate = 1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer

        