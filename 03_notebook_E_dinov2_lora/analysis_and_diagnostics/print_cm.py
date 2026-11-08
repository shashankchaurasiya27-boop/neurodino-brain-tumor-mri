import numpy as np
from pathlib import Path
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F

EXP_DIR = Path(sorted(glob.glob("../outputs/EXP_*"), reverse=True)[0])
npz_data = np.load(EXP_DIR / 'features/dataset_features.npz', allow_pickle=True)
train_feat_all = npz_data['train_feat']
train_lbl_all = npz_data['train_lbl']
test_feat = npz_data['test_feat']
test_lbl = npz_data['test_lbl']

classes = sorted(np.unique(train_lbl_all))
class_to_idx = {c: i for i, c in enumerate(classes)}
train_lbl_int = np.array([class_to_idx[l] for l in train_lbl_all])
test_lbl_int = np.array([class_to_idx[l] for l in test_lbl])

class ContrastiveAdapter(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim)
        )
    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)

device = torch.device("cpu")
adapter = ContrastiveAdapter().to(device)
adapter.load_state_dict(torch.load(EXP_DIR / 'checkpoints/contrastive_adapter_best.pth', map_location=device))
adapter.eval()

with torch.no_grad():
    new_train_feat = adapter(torch.FloatTensor(train_feat_all)).numpy()
    new_test_feat = adapter(torch.FloatTensor(test_feat)).numpy()

clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
clf.fit(new_train_feat, train_lbl_int)
preds = clf.predict(new_test_feat)

cm = confusion_matrix(test_lbl_int, preds)
print("Classes:", classes)
print("Confusion Matrix:\\n", cm)
