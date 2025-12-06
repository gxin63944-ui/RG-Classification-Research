import torch
import torch.nn as nn
import torch.optim as optim
from model import get_resnet101
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import random
import os
import time

CONFIG = {
    'seed': 14,            
    'num_classes': 2,           
    'batch_size': 32,        
    'learning_rate': 1e-3,     
    'weight_decay': 1e-4,     
    'num_epochs': 500,         
    'k_folds': 5,              
    'patience': 20,            
    'img_size': 256,            
    'data_dir': '/media/xsar/F/resnet/data', 
    'model_save_dir': '/media/xsar/F/resnet/data/saved_models'
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def main():
    # 初始化
    set_seed(CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(CONFIG['model_save_dir'], exist_ok=True)
    
    print(f"Starting Training on {device}...")
    print(f"Config: ResNet-101 | Batch: {CONFIG['batch_size']} | Epochs: {CONFIG['num_epochs']}")

    # ---------------------------------------------------------
    # Data Augmentation
    # ---------------------------------------------------------
    train_transforms = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    if not os.path.exists(CONFIG['data_dir']):
        print(f"Error: Data directory '{CONFIG['data_dir']}' not found.")
        return

    full_dataset = datasets.ImageFolder(CONFIG['data_dir'])
    targets = full_dataset.targets 
    class_names = full_dataset.classes 
    

    skf = StratifiedKFold(n_splits=CONFIG['k_folds'], shuffle=True, random_state=CONFIG['seed'])
    
    # 存储每一折的最佳结果
    fold_metrics = {
        'accuracy': [],
        'precision': [], 
        'recall': [],    
        'f1': [],       
        'time': []
    }

    total_start_time = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\n{'='*20} Fold {fold+1}/{CONFIG['k_folds']} {'='*20}")
        fold_start_time = time.time()


        train_subset_raw = Subset(full_dataset, train_idx)
        val_subset_raw = Subset(full_dataset, val_idx)
        
        train_dataset = TransformedSubset(train_subset_raw, transform=train_transforms)
        val_dataset = TransformedSubset(val_subset_raw, transform=val_transforms)

        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)

 
        model = get_resnet101(num_classes=CONFIG['num_classes'], pretrained=False)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


        best_acc = 0.0
        best_metrics = {} # 存储最佳模型对应的所有详细指标
        epochs_no_improve = 0
        
        for epoch in range(CONFIG['num_epochs']):
            # Train
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # 计算指标
            val_acc = accuracy_score(all_labels, all_preds)
            scheduler.step()

            if val_acc > best_acc:
                best_acc = val_acc
                epochs_no_improve = 0
                
                p_class, r_class, f1_class, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
                p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
                
                best_metrics = {
                    'accuracy': val_acc,
                    'precision_macro': p_macro,
                    'recall_macro': r_macro,
                    'f1_macro': f1_macro,
                    'per_class_precision': p_class,
                    'per_class_recall': r_class,
                    'per_class_f1': f1_class
                }
                
                # 保存模型
                save_path = os.path.join(CONFIG['model_save_dir'], f'resnet101_fold{fold+1}_best.pth')
                torch.save(model.state_dict(), save_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= CONFIG['patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        fold_end_time = time.time()
        fold_duration = fold_end_time - fold_start_time
        
        # 记录每折的最佳结果
        fold_metrics['accuracy'].append(best_metrics['accuracy'])
        fold_metrics['precision'].append(best_metrics['precision_macro'])
        fold_metrics['recall'].append(best_metrics['recall_macro'])
        fold_metrics['f1'].append(best_metrics['f1_macro'])
        fold_metrics['time'].append(fold_duration)

        print(f"\n[Fold {fold+1} Report]")
        print(f"Training Time: {fold_duration:.2f} seconds")
        print(f"Best Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"Macro F1-Score: {best_metrics['f1_macro']:.4f}")
        print("-" * 40)
        print(f"{'Class':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
        print("-" * 40)
        for i, class_name in enumerate(class_names):
            print(f"{class_name:<10} | {best_metrics['per_class_precision'][i]:.4f}     | {best_metrics['per_class_recall'][i]:.4f}     | {best_metrics['per_class_f1'][i]:.4f}")
        print("-" * 40)

    # ---------------------------------------------------------
    # 所有折的平均结果
    # ---------------------------------------------------------
    print(f"\n{'='*20} Final Summary (Average over 5 Folds) {'='*20}")
    print(f"Total Training Time: {time.time() - total_start_time:.2f} seconds")
    print(f"Average Accuracy:  {np.mean(fold_metrics['accuracy']):.4f} (±{np.std(fold_metrics['accuracy']):.4f})")
    print(f"Average Macro F1:  {np.mean(fold_metrics['f1']):.4f}")
    print(f"Average Precision: {np.mean(fold_metrics['precision']):.4f}")
    print(f"Average Recall:    {np.mean(fold_metrics['recall']):.4f}")
    print(f"Average Time/Fold: {np.mean(fold_metrics['time']):.2f} seconds")

if __name__ == '__main__':
    main()