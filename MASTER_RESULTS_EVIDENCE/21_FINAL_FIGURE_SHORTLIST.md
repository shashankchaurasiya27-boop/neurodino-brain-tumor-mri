# Final Manuscript Figure Shortlist & Recommendations

We recommend a focused set of **4 high-impact figures** for the final paper:

1. **Figure 1: Tri-Panel Confusion Matrix & Error Shift (`fig1_confusion_matrices.png`)**
   * *Type*: Heatmap Tri-Panel (Frozen Control, Controlled LoRA, Difference Matrix)
   * *Source*: `outputs/EXP_LORA_CONTROLLED_20260811_211611/figures/confusion_matrices.png`
   * *Purpose*: Proves class-wise accuracy improvements and shows the net reduction of Glioma-Meningioma confusion.

2. **Figure 2: Representation Space 2D UMAP Projection (`fig2_umap_2d_projection.png`)**
   * *Type*: 2D Scatter Projection (Frozen DINOv2 vs Controlled LoRA)
   * *Source*: `outputs/EXP_LORA_CONTROLLED_20260811_211611/figures/umap_2d_projection.png`
   * *Purpose*: Demonstrates the geometric densification and cluster separation achieved in the 768-D feature space.

3. **Figure 3: Representation Ceiling vs Backbone Adaptation (`fig3_representation_ceiling.png`)**
   * *Type*: Horizontal Bar Benchmark Chart
   * *Source*: `MASTER_RESULTS_EVIDENCE/verified_figures/fig3_representation_ceiling.png`
   * *Purpose*: Contrasts 7 frozen downstream classifiers (plateauing at 93.52%) against the LoRA-adapted backbone (95.21%).

4. **Figure 4: Data Curation & Duplicate Isolation Flowchart (`fig4_data_curation_flow.png`)**
   * *Type*: Methodological Flow Diagram
   * *Source*: `MASTER_RESULTS_EVIDENCE/verified_figures/fig4_data_curation_flow.png`
   * *Purpose*: Details the 7,200 -> 6,741 deterministic cleaning pipeline guaranteeing zero leakage.
