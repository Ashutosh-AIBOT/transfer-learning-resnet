import torch
import torch.nn as nn
from tqdm import tqdm
import time

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=20):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        train_loss, train_corrects = 0.0, 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)
            
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_corrects.double() / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_corrects.double() / len(val_loader.dataset)
        
        if scheduler:
            scheduler.step()
            
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc.item())
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc.item())
        
        duration = time.time() - t0
        print(f"Epoch {epoch+1}: T-Loss: {epoch_train_loss:.4f} T-Acc: {epoch_train_acc:.4f} | V-Loss: {epoch_val_loss:.4f} V-Acc: {epoch_val_acc:.4f} ({duration:.1f}s)")
        
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            
    return model, history
