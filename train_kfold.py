import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset, Subset

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent  # CSV sits next to this script


# ===============================
# 1D CNN Model Definition
# ===============================
class Simple1DCNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 1209, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 6),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ===============================
# Dataset Loader
# ===============================
def load_data_set(filename):
    data = np.loadtxt(filename, delimiter=",")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)

# ===============================
# Training Function for Single Fold
# ===============================
def train_fold(train_dataset, val_dataset, fold_num, device='cpu'):
    print(f"\n{'='*50}")
    print(f"Training Fold {fold_num}")
    print(f"{'='*50}")
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    model = Simple1DCNN1().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3, factor=0.5)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Early stopping setup
    best_val_loss = float('inf')
    patience = 6
    patience_counter = 0
    best_model_state = None

    num_epochs = 40
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        train_acc = 100 * correct / total
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(train_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)

                val_loss += loss.item()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc)

        if epoch % 5 == 0:  # Print every 5 epochs to reduce clutter
            print(f"Epoch {epoch + 1}/{num_epochs} — Train Loss: {train_losses[-1]:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_losses[-1]:.4f}, Acc: {val_acc:.2f}%")
        
        scheduler.step(val_losses[-1])

        # Early stopping
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_counter = 0
            best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"Fold {fold_num} - Best Val Loss: {best_val_loss:.4f}, Final Val Acc: {val_accuracies[np.argmin(val_losses)]:.2f}%")
    
    return model, train_losses, val_losses, train_accuracies, val_accuracies, best_val_loss

# ===============================
# Evaluation Function
# ===============================
def evaluate_fold(model, test_dataset, fold_num, device='cpu'):
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    total = 0
    correct = 0
    all_labels = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    acc = 100 * correct / total
    print(f"Fold {fold_num} Test Accuracy: {acc:.2f}%")
    
    return acc, all_labels, all_preds

# ===============================
# Main K-Fold Cross-Validation Function
# ===============================
def run_kfold_cv(k_folds=5):
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    xdataset, ydataset = load_data_set(DATA_DIR / "numbersSoftMediumHardBigAirTapsBiggerTapupAir.csv")
    xdataset = np.array([x[:4836] for x in xdataset])  # Truncate

    # EDA: Plot sample from each class
    plt.figure(figsize=(12, 4))
    for cls in np.unique(ydataset):
        plt.plot(xdataset[ydataset == cls][0], label=f"class {cls}")
    plt.legend()
    plt.title("Sample Sequences per Class")
    plt.show()

    # EDA: Class distribution
    class_counts = np.bincount(ydataset)
    plt.figure(figsize=(8, 6))
    plt.bar(np.arange(len(class_counts)), class_counts)
    plt.title("Number of Samples per Class")
    plt.xlabel("Class")
    plt.ylabel("Count")
    for i, count in enumerate(class_counts):
        plt.text(i, count + 0.1, str(count), ha='center')
    plt.show()

    # Convert to tensors
    X_tensor = torch.FloatTensor(xdataset).unsqueeze(1)
    y_tensor = torch.LongTensor(ydataset)
    full_dataset = TensorDataset(X_tensor, y_tensor)
    
    print(f"\nTotal trainable parameters: {sum(p.numel() for p in Simple1DCNN1().parameters()):,}")
    print(f"Dataset size: {len(full_dataset)}")
    print(f"Running {k_folds}-fold cross-validation")

    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Store results for each fold
    fold_results = {
        'test_accuracies': [],
        'val_losses': [],
        'all_labels': [],
        'all_preds': [],
        'train_histories': []
    }
    
    # Perform k-fold cross-validation
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(xdataset, ydataset), 1):
        # Further split train_val into train and validation (80-20 split of train_val)
        train_val_labels = ydataset[train_val_idx]
        
        # Stratified split for train/val within the fold
        np.random.seed(42 + fold)  # Different seed per fold
        train_idx = []
        val_idx = []
        
        for cls in np.unique(train_val_labels):
            cls_indices = train_val_idx[train_val_labels == cls]
            np.random.shuffle(cls_indices)
            n_train = int(0.8 * len(cls_indices))
            train_idx.extend(cls_indices[:n_train])
            val_idx.extend(cls_indices[n_train:])
        
        # Create datasets for this fold
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)
        test_dataset = Subset(full_dataset, test_idx)
        
        print(f"\nFold {fold}: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        # Train model for this fold
        model, train_losses, val_losses, train_accs, val_accs, best_val_loss = train_fold(
            train_dataset, val_dataset, fold, device
        )
        
        # Evaluate on test set
        test_acc, labels, preds = evaluate_fold(model, test_dataset, fold, device)
        
        # Store results
        fold_results['test_accuracies'].append(test_acc)
        fold_results['val_losses'].append(best_val_loss)
        fold_results['all_labels'].extend(labels)
        fold_results['all_preds'].extend(preds)
        fold_results['train_histories'].append({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        })
    
    # Calculate and display overall results
    mean_acc = np.mean(fold_results['test_accuracies'])
    std_acc = np.std(fold_results['test_accuracies'])
    
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION RESULTS ({k_folds}-fold)")
    print(f"{'='*60}")
    print(f"Test Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Individual fold accuracies: {[f'{acc:.2f}%' for acc in fold_results['test_accuracies']]}")
    print(f"Best fold: {np.max(fold_results['test_accuracies']):.2f}%")
    print(f"Worst fold: {np.min(fold_results['test_accuracies']):.2f}%")
    
    # Overall classification report and confusion matrix
    print(f"\nOverall Classification Report:")
    print(classification_report(fold_results['all_labels'], fold_results['all_preds']))
    
    # Plot overall confusion matrix
    matrix = confusion_matrix(fold_results['all_labels'], fold_results['all_preds'])
    plt.figure(figsize=(8, 6))
    seaborn.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=np.unique(fold_results['all_labels']), 
                    yticklabels=np.unique(fold_results['all_labels']))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Overall Confusion Matrix (All Folds)")
    plt.show()
    
    # Plot training curves for all folds
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    for i, history in enumerate(fold_results['train_histories']):
        plt.plot(history['train_losses'], alpha=0.7, label=f'Fold {i+1}')
    plt.title('Training Loss Across Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    for i, history in enumerate(fold_results['train_histories']):
        plt.plot(history['val_losses'], alpha=0.7, label=f'Fold {i+1}')
    plt.title('Validation Loss Across Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.bar(range(1, k_folds+1), fold_results['test_accuracies'])
    plt.title('Test Accuracy by Fold')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    for i, acc in enumerate(fold_results['test_accuracies']):
        plt.text(i+1, acc + 1, f'{acc:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    return fold_results

# ===============================
# Run K-Fold Cross-Validation
# ===============================
if __name__ == "__main__":
    results = run_kfold_cv(k_folds=5)