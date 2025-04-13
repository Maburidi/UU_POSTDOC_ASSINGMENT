from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
import torch.optim as optim                         
from torchvision import transforms             
from DataLoader0 import ImageDataset              
from simpleCNNmodel import SimpleCNN            


#------------------- Transforms: Define augmentation chain ----------------------

transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(30, 30), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    #transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip horizontally
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = ImageDataset('/content/data0/training', transform=transform)    
                                                                                
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)           



k_folds = 5
num_epochs = 100


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    torch.manual_seed(42)
    dataset = train_dataset  

    kfold = KFold(n_splits=k_folds, shuffle=True)

    print('--------------------------------------')

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('----------------------------------------')

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        train_loader = DataLoader(dataset, batch_size=10, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=10, sampler=val_subsampler)

        model = SimpleCNN().to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(1, num_epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                # Update running loss and accuracy
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            print(f'Epoch {epoch}, Loss: {running_loss/len(train_loader)}, Training Accuracy: {100 * correct / total}%')

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        print(f'Validation Accuracy for fold {fold}: {100 * correct / total}%')
        print('--------------------------------')
    torch.save(model.state_dict(), '/content/model_trained.pth')




if __name__ == "__main__":
    main()
