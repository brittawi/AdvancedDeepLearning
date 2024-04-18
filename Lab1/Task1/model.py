import torch.nn as nn
import lightning as L
import torch.nn.functional as F
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)

        return out
    
class LitNet(L.LightningModule):
    
    def __init__(self, model, lr = 0.0001):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        words, labels = batch
        
        # Forward pass
        outputs = self.model(words)
        loss = F.cross_entropy(outputs, labels)
        
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        
        return loss
    
    def evaluate(self, batch, stage=None):
        words, labels = batch
        
        # Forward pass
        outputs = self.model(words)
        loss = F.cross_entropy(outputs, labels)
        
        _,preds = torch.max(outputs,dim=1)
        acc = accuracy_score(preds, labels)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")
    
    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)






