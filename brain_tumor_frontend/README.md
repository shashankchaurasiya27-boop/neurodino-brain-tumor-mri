# NeuroDINO® | Brain Tumor MRI Diagnostic & Representation Research Platform

A modern, high-performance web application designed to demonstrate parameter-efficient representation adaptation of Vision Foundation Models (DINOv2 ViT-B/14 + QKV-LoRA) for four-class brain tumor MRI diagnosis.

This frontend application is completely decoupled from the core research codebase and lives outside the research folder at `/Users/shashankchaurasia/Documents/brain_tumor_frontend`, linking dynamically to the certified checkpoints and assets in `/Users/shashankchaurasia/Documents/tumor copy` without modifying any research files.

---

## 🎨 Design System Implementation

Faithfully constructed according to the design specification:
- **Typography**: `Plus Jakarta Sans` / `Inter` / `Labil Grotesk` with geometric hierarchy (`Display 2XL` to `Body M`).
- **Color Palette**:
  - **Primary**: `#9B43F3` (Vibrant Electric Violet)
  - **Alerts**: Success `#29AE29` (Healthy Control), Warning `#FFC62A` (Meningioma), Error `#EF4444` (Glioma / Tumor Alert)
  - **Accents**: Lavender `#6E3CF5`, Sussie (Cyan) `#38E1F1`, Blue Jeans `#0082FF`
- **Component Language**:
  - **Floating Pill Navbar**: High-contrast, glassmorphism rounded-full menu.
  - **Purple Mockup Card Container**: Device-frame showcase plates matching Joey® / Djectstudio design language.
  - **Interactive Inference Studio**: Instant drag-and-drop file upload + side-by-side comparative diagnostics (Controlled LoRA vs Frozen Baseline).
  - **Accordion System**: Smooth animated expandable cards with `+` / `−` toggles for architecture and scientific methodology FAQs.
  - **Footer with Pill CTA**: High-impact editorial "Let's talk!" footer with clinical disclaimers.

---

## 🚀 Quick Start

To launch the web platform:

```bash
cd /Users/shashankchaurasia/Documents/brain_tumor_frontend
./run_server.sh
```

Or using python directly:

```bash
/Users/shashankchaurasia/miniconda3/bin/python app.py
```

Then open your browser at **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.

---

## 🔬 Core Scientific Capabilities

1. **Dual Side-by-Side Model Comparison**:
   - **Arm B (Adapted)**: DINOv2 ViT-B/14 + PEFT QKV-LoRA ($r=8, \alpha=16$) + 3-layer MLP head (95.21% test accuracy).
   - **Arm A (Frozen Baseline)**: DINOv2 ViT-B/14 + bit-identical initial MLP head (92.11% test accuracy).
2. **Sub-30ms Real-Time Inference**:
   - Warm inference latency: **~27.9 ms** on Apple Silicon Metal Performance Shaders (MPS).
3. **50-Image Verified Showcase Gallery**:
   - Filterable across Glioma (13), Meningioma (13), Healthy Normal (12), and Pituitary (12).
   - One-click live evaluation against certified ground-truth test data.
4. **LoRA Error Rescue Alerts**:
   - Automatically highlights borderline scans where QKV-LoRA representation adaptation resolved baseline misclassifications.
