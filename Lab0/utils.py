import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import numpy as np


def train_model(model, criterion, optimizer, train_loader, val_loader, file_name, writer, num_epochs = 10, dev = "cpu", show_plot = True):
    
    min_loss = 10000
    
    # Track loss
    training_loss, validation_loss = [], []
    
    # Track accuracy
    training_acc, validation_acc = [], []
    
    for i in range(num_epochs):
        # Track loss
        epoch_training_loss, epoch_validation_loss = 0, 0
        train_size, val_size = 0, 0
        
        # track accuracy
        train_correct, val_correct = 0, 0

        # training
        model.train(True)
        for batch_nr, (data, labels) in enumerate(train_loader):
            
            data = data.to(dev)
            labels = labels.to(dev)
            
            # predict
            pred = model(data)

            # calculate accuracy
            _,preds = torch.max(pred,dim=1)
            train_correct += torch.sum(preds==labels).item()
            
            # Clear stored gradient values
            optimizer.zero_grad()
            
            loss = criterion(pred, labels)
            
            # Backpropagate the loss through the network to find the gradients of all parameters
            loss.backward()
            
            # Update the parameters along their gradients
            optimizer.step()
            
            # Update loss
            epoch_training_loss += loss.cpu().detach().numpy()
            
            train_size += len(data)
            
        # validation
        model.eval()
        for batch_nr, (data, labels) in enumerate(val_loader):
            
            data = data.to(dev)
            labels = labels.to(dev)
            
            # predict
            pred = model(data)
            
            # calculate accuracy
            _,preds = torch.max(pred,dim=1)
            val_correct += torch.sum(preds==labels).item()
             
            # calculate loss
            loss = criterion(pred, labels)
            
            # Update loss
            epoch_validation_loss += loss.cpu().detach().numpy()
            val_size += len(data)
            
        # check if loss is smaller than before for each epoch, if so safe model
        if (epoch_validation_loss/val_size)<min_loss:
            torch.save(model, file_name)
            min_loss = loss
            
        # Save loss for plot
        training_loss.append(epoch_training_loss/(train_size))
        writer.add_scalar("Loss/train", epoch_training_loss/(train_size), i)
        validation_loss.append(epoch_validation_loss/val_size)
        writer.add_scalar("Loss/validation", epoch_validation_loss/val_size, i)
        
        # Save accuracy for plot
        training_acc.append(train_correct/(train_size))
        validation_acc.append(val_correct/val_size)

        # Print loss every 5 epochs
        if i % 5 == 0:
            print(f'Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}')
            print(f'Train accuracy = {train_correct/(train_size)}')
            print(f'Validation accuracy = {val_correct/val_size}')
        
    if show_plot:
        # Plot training and validation loss
        epoch = np.arange(len(training_loss))
        plt.figure(figsize=(8,4), dpi=100)
        plt.plot(epoch, training_loss, 'r', label='Training loss',)
        plt.plot(epoch, validation_loss, 'b', label='Validation loss')
        plt.legend()
        plt.xlabel('Epoch'), plt.ylabel('Loss')
        plt.show()
        
        # Plot training and validation accuracy
        plt.figure(figsize=(8,4), dpi=100)
        plt.plot(epoch, training_acc, 'r', label='Training accuracy',)
        plt.plot(epoch, validation_acc, 'b', label='Validation accuracy')
        plt.legend()
        plt.xlabel('Epoch'), plt.ylabel('Accuracy')
        plt.show()
        
    idx = np.argmin(validation_loss)
    print(f'lowest loss for validation set: {np.min(validation_loss)}, with an accuracy of {validation_acc[idx]}')
    
    writer.flush()
    
    
def test_model(model, test_loader, dev = "cpu"):

    test_acc = 0
    y_pred, y_true = [], []
    test_size = 0

    model.eval()
    with torch.no_grad():
        for batch_nr, (data, labels) in enumerate(test_loader):
        
            data = data.to(dev)
            labels = labels.to(dev)
            
            # predict
            pred = model(data)
                
            # calculate accuracy
            _,preds = torch.max(pred,dim=1)
            test_acc += torch.sum(preds==labels).item()
        
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            
            test_size += len(data)
    
    test_acc /= test_size / 100

    print(f"Test accuracy is {np.round(test_acc)}%.") 
    
    # Confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cf_matrix)
    cm_display.plot(colorbar=False)
    plt.show()
    
    