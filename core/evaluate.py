import torch
import os
from torch.utils.data import DataLoader
from model import build_model
from preprocess import ASVDataset

# Setup hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Dynamic Path Handling (Look up one level from core/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ASVspoof 2019 Dev Set Paths (Using the 'data' folder at root)
dev_protocol = os.path.join(BASE_DIR, "data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt")
dev_data = os.path.join(BASE_DIR, "data/LA/ASVspoof2019_LA_dev/flac")
model_path = os.path.join(BASE_DIR, "deepfake_detector.pth")

def run_evaluation():
    # 2. Load the architecture and the saved weights
    model = build_model(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded trained weights from {model_path}")
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Run train.py first!")
        return

    model.eval()

    # 3. Load the Dev subset (A07-A19 attacks for LA)
    dev_dataset = ASVDataset(dev_protocol, dev_data, subset_size=500)
    dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)

    correct = 0
    total = 0

    # 4. Validation Loop (No gradients needed)
    print("Evaluating on dev set...")
    with torch.no_grad():
        for specs, labels in dev_loader:
            specs, labels = specs.to(device), labels.to(device)
            outputs = model(specs)
            
            # Convert logits to binary predictions (0 or 1)
            # probabilities > 0.5 are 'spoof' (1), <= 0.5 are 'bonafide' (0) [cite: 5, 11]
            predictions = (torch.sigmoid(outputs) > 0.5).int().squeeze()
            
            # Match prediction to ground truth labels [cite: 5, 10]
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\n--- Evaluation Results ---")
    print(f"Validation Accuracy: {accuracy:.2f}%")
    print(f"Samples Tested: {total}")

if __name__ == "__main__":
    run_evaluation()