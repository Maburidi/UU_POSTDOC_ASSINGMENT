import torch.nn as nn
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns                                                  
import matplotlib.pyplot as plt    
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

                    
model = SimpleCNN()  
model.load_state_dict(torch.load('/content/model_trained.pth'))     
model.eval()                   



transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(30, 30), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    #transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip horizontally
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_dataset = ImageDataset('/content/data0/testing',   transform=transform)    
                                                                                
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)            

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()

total = 0
correct = 0
test_loss = 0.0

y_true = []
y_scores = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]  
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        y_true.extend(labels.cpu().numpy())
        y_scores.extend(probabilities.cpu().numpy())

average_loss = test_loss / total
accuracy = 100 * correct / total

print(f'Test Loss: {average_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
