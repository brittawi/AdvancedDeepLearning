import torch
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

import json
import random
import sys

from transformer_model import TransformerModel

sys.path.insert(0, 'C:/awilde/britta/LTU/AdvancedDeepLearning/Labs/Lab1/Task1')

from data_loading_code import getData, processUserInput
from model import NeuralNet, LitNet

# hyperparameters
EPOCHS = 5
BATCH_SIZE = 20
LEARNING_RATE = 0.001
HIDDEN_SIZE = 8
NUM_CLASSES = 2 # we only have two classes
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 2  # number of heads in ``nn.MultiheadAttention``
dropout = 0.2  # dropout probability

TRAIN = True

def main(args = None):
    
    # get the preprocessed Data
    x_train, y_train, x_val, y_val, vocabsize, word_vectorizer = getData("C:/awilde/britta/LTU/AdvancedDeepLearning/Labs/Lab1/Task1/amazon_cells_labelled.txt")
    x_train = x_train.to(torch.int64)
    x_val = x_val.to(torch.int64)
    
    # train the model
    if TRAIN:
        
        # put data into Dataloaders
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=7, persistent_workers=True)
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=BATCH_SIZE, shuffle=False, num_workers=7, persistent_workers=True)
    
        # initialise neural Net
        model = TransformerModel(vocabsize, emsize, nhead, d_hid, nlayers, dropout)
        litModel = LitNet(model)

        # initialise logger
        logger = TensorBoardLogger("tb_logs", name="transformerModel")

        # intialise Trainer
        trainer = L.Trainer(max_epochs=EPOCHS, 
                            callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
                            logger=logger,
                            fast_dev_run=True)

        # train the model
        trainer.fit(litModel, train_dataloaders=train_loader, val_dataloaders=val_loader)
        # save best model
        trainer.save_checkpoint("best_model_transfomer.ckpt")
    
    # load model
    model = TransformerModel(vocabsize, emsize, nhead, d_hid, nlayers, dropout)
    litModel = LitNet.load_from_checkpoint("best_model_transformer.ckpt", model=model)

    # disable randomness, dropout, etc...
    model.eval()
    
    # # load response data for chatBot
    # with open('responses.json', 'r') as json_data:
    #     responses = json.load(json_data)
    
    # # chatBot functionality
    # bot_name = "Bot"
    # print("Let's chat! (type 'quit' to exit)")
    # while True:
        
    #     sentence = input("You: ")
    #     if sentence == "quit":
    #         break
        
    #     # process and predict for the user input
    #     userInput = processUserInput(sentence, word_vectorizer)
    #     output = model(userInput)
        
    #     _,pred = torch.max(output,dim=1)
    #     pred = pred.item()

    #     # select random entry from list for the prediction
    #     resp = random.choice(responses["responses"][pred]["answers"])
    #     print(f"{bot_name}: {resp}")

if __name__ == '__main__':
    
    main()