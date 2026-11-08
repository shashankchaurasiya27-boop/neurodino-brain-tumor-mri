import matplotlib.pyplot as plt
import numpy as np

models = [
    'Nearest Centroid (Frozen)',
    'kNN (k=5) (Frozen)',
    'Linear SVM (Frozen)',
    'Ridge Classifier (Frozen)',
    'Matched 3-Layer MLP (Arm A Control)',
    'Logistic Regression (Frozen Ceiling)',
    'Controlled QKV-LoRA DINOv2 (Arm B)'
]
accuracies = [70.40, 86.00, 91.30, 91.30, 92.11, 93.52, 95.21]
colors = ['#CBD5E0', '#CBD5E0', '#CBD5E0', '#CBD5E0', '#4A5568', '#2B6CB0', '#2C7A7B']

plt.figure(figsize=(10, 5.5), dpi=300)
bars = plt.barh(models, accuracies, color=colors, height=0.65)
plt.xlim(65, 100)
plt.xlabel('Test Accuracy (%)', fontsize=12, fontweight='bold', labelpad=8)
plt.title('Downstream Representation Capacity vs. Backbone LoRA Adaptation', fontsize=13, fontweight='bold', pad=12)
plt.axvline(93.52, color='#E53E3E', linestyle='--', linewidth=1.5, label='Frozen Linear Probe Ceiling (93.52%)')

for bar, acc in zip(bars, accuracies):
    plt.text(acc + 0.4, bar.get_y() + bar.get_height()/2, f'{acc:.2f}%', va='center', ha='left', fontsize=10.5, fontweight='bold')

plt.legend(loc='lower right', frameon=True, fontsize=10.5)
plt.grid(axis='x', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('verified_figures/fig3_representation_ceiling.png')
plt.savefig('verified_figures/fig3_representation_ceiling.pdf')
plt.close()
