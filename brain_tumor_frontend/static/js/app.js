/**
 * NEURODINO® Brain Tumor MRI Platform - Complete Interactive Frontend Logic
 * Supports dual-model live inference, radiologist darkroom viewer,
 * storytelling genesis timeline, latent PCA/UMAP explorer, and persistent session history.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Navigation & Theme
  const navPillItems = document.querySelectorAll(".nav-pill-btn");
  const historyCounter = document.getElementById("history-counter");
  const themeToggle = document.getElementById("theme-toggle");
  const themeIcon = document.getElementById("theme-icon");
  const themeLabel = document.getElementById("theme-label");
  const drawerThemeToggle = document.getElementById("drawer-theme-toggle");
  const drawerThemeIcon = document.getElementById("drawer-theme-icon");
  const drawerThemeLabel = document.getElementById("drawer-theme-label");
  const mobileToggle = document.getElementById("mobile-toggle");
  const mobileDrawer = document.getElementById("mobile-drawer");
  const drawerClose = document.getElementById("drawer-close");
  const drawerLinks = document.querySelectorAll(".drawer-link");

  // Theme Management (Supports Figma Light & Dark Modes)
  function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("neurodino_theme", theme);
    const isLight = theme === "light";
    if (themeIcon) themeIcon.textContent = isLight ? "🌙" : "☀️";
    if (themeLabel) themeLabel.textContent = isLight ? "Dark Mode" : "Light Mode";
    if (drawerThemeIcon) drawerThemeIcon.textContent = isLight ? "🌙" : "☀️";
    if (drawerThemeLabel) drawerThemeLabel.textContent = isLight ? "Switch to Dark Mode" : "Switch to Light Mode";
  }

  const savedTheme = localStorage.getItem("neurodino_theme") || "dark";
  setTheme(savedTheme);

  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme") || "dark";
      setTheme(current === "dark" ? "light" : "dark");
    });
  }

  if (drawerThemeToggle) {
    drawerThemeToggle.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme") || "dark";
      setTheme(current === "dark" ? "light" : "dark");
    });
  }

  // Mobile Drawer Toggle
  if (mobileToggle && mobileDrawer) {
    mobileToggle.addEventListener("click", () => {
      mobileDrawer.classList.add("open");
      document.body.style.overflow = "hidden";
    });
  }

  if (drawerClose && mobileDrawer) {
    drawerClose.addEventListener("click", () => {
      mobileDrawer.classList.remove("open");
      document.body.style.overflow = "";
    });
  }

  drawerLinks.forEach((link) => {
    link.addEventListener("click", () => {
      if (mobileDrawer) {
        mobileDrawer.classList.remove("open");
        document.body.style.overflow = "";
      }
    });
  });

  // Studio Elements
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const previewContainer = document.getElementById("preview-container");
  const previewImage = document.getElementById("preview-image");
  const previewClose = document.getElementById("preview-close");
  const btnAnalyze = document.getElementById("btn-analyze");
  const analyzeBtnText = document.getElementById("analyze-btn-text");
  const analyzeSpinner = document.getElementById("analyze-spinner");

  // Radiologist Darkroom Controls
  const sliderBrightness = document.getElementById("slider-brightness");
  const sliderContrast = document.getElementById("slider-contrast");
  const valBrightness = document.getElementById("val-brightness");
  const valContrast = document.getElementById("val-contrast");
  const btnInvert = document.getElementById("btn-invert");
  const btnResetDarkroom = document.getElementById("btn-reset-darkroom");
  const loupeLens = document.getElementById("loupe-lens");
  const loupeWrapper = document.getElementById("loupe-wrapper");

  // Results Elements
  const resultsContainer = document.getElementById("results-container");
  const emptyState = document.getElementById("empty-state");
  const loraClass = document.getElementById("lora-class");
  const loraPathology = document.getElementById("lora-pathology");
  const loraConfidence = document.getElementById("lora-confidence");
  const loraLatency = document.getElementById("lora-latency");
  const loraProbBars = document.getElementById("lora-prob-bars");

  const frozenClass = document.getElementById("frozen-class");
  const frozenPathology = document.getElementById("frozen-pathology");
  const frozenConfidence = document.getElementById("frozen-confidence");
  const frozenLatency = document.getElementById("frozen-latency");
  const frozenProbBars = document.getElementById("frozen-prob-bars");
  const rescuedAlert = document.getElementById("rescued-alert");

  // Showcase & Story Elements
  const showcaseGrid = document.getElementById("showcase-grid");
  const filterPills = document.querySelectorAll(".filter-pill");
  const storyTimelineContainer = document.getElementById("story-timeline-container");
  const geomTableBody = document.getElementById("geom-table-body");

  // Results & Latent Projection Elements
  const projectionImg = document.getElementById("projection-img");
  const projectionTitle = document.getElementById("projection-title");
  const projectionDesc = document.getElementById("projection-desc");
  const resultsTabBtns = document.querySelectorAll(".results-tab-btn");

  // History Elements
  const historyTableBody = document.getElementById("history-table-body");
  const btnExportCsv = document.getElementById("btn-export-csv");
  const btnClearHistory = document.getElementById("btn-clear-history");
  const btnPrintReport = document.getElementById("btn-print-report");

  let currentFile = null;
  let currentImageSrc = "";
  let allShowcaseScans = [];
  let isGrayscaleInverted = false;
  let currentLastResult = null;

  const CLASS_COLORS = {
    glioma: "#EC4899",     // Pink/Rose
    meningioma: "#F59E0B", // Amber/Gold
    notumor: "#29AE29",    // Vivid Green
    pituitary: "#9B43F3"   // Electric Violet
  };

  const CLASS_LABELS = {
    glioma: "Glioma Tumor",
    meningioma: "Meningioma Tumor",
    notumor: "Healthy Brain (No Tumor)",
    pituitary: "Pituitary Adenoma"
  };

  // ==========================================================================
  // FLOATING PILL NAVBAR SCROLL SYNC
  // ==========================================================================
  window.addEventListener("scroll", () => {
    const scrollPos = window.scrollY + 200;
    const sections = document.querySelectorAll("section[id]");

    sections.forEach((sec) => {
      const top = sec.offsetTop;
      const height = sec.offsetHeight;
      const id = sec.getAttribute("id");

      if (scrollPos >= top && scrollPos < top + height) {
        navPillItems.forEach((pill) => {
          pill.classList.remove("active");
          if (pill.getAttribute("href") === `#${id}`) {
            pill.classList.add("active");
          }
        });
      }
    });
  });

  // ==========================================================================
  // RADIOLOGIST DARKROOM VIEWER CONTROLS (Brightness, Contrast, Invert, Loupe)
  // ==========================================================================
  function applyDarkroomFilters() {
    const b = sliderBrightness.value;
    const c = sliderContrast.value;
    const inv = isGrayscaleInverted ? 1 : 0;
    valBrightness.textContent = `${b}%`;
    valContrast.textContent = `${c}%`;
    previewImage.style.filter = `brightness(${b}%) contrast(${c}%) invert(${inv})`;
  }

  sliderBrightness.addEventListener("input", applyDarkroomFilters);
  sliderContrast.addEventListener("input", applyDarkroomFilters);

  btnInvert.addEventListener("click", () => {
    isGrayscaleInverted = !isGrayscaleInverted;
    btnInvert.classList.toggle("active", isGrayscaleInverted);
    applyDarkroomFilters();
  });

  btnResetDarkroom.addEventListener("click", () => {
    sliderBrightness.value = 100;
    sliderContrast.value = 100;
    isGrayscaleInverted = false;
    btnInvert.classList.remove("active");
    applyDarkroomFilters();
  });

  // Interactive 2.5x Magnifier Loupe (Mouse & Touch Enabled)
  function updateLoupe(clientX, clientY) {
    if (!previewImage.src || previewContainer.style.display === "none") return;
    loupeLens.style.display = "block";

    const rect = previewImage.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;

    if (x < 0 || x > rect.width || y < 0 || y > rect.height) {
      loupeLens.style.display = "none";
      return;
    }

    const lensRadius = 65;
    loupeLens.style.left = `${x - lensRadius}px`;
    loupeLens.style.top = `${y - lensRadius}px`;

    const zoom = 2.5;
    loupeLens.style.backgroundImage = `url('${previewImage.src}')`;
    loupeLens.style.backgroundSize = `${rect.width * zoom}px ${rect.height * zoom}px`;
    loupeLens.style.backgroundPosition = `-${x * zoom - lensRadius}px -${y * zoom - lensRadius}px`;
  }

  loupeWrapper.addEventListener("mousemove", (e) => updateLoupe(e.clientX, e.clientY));
  loupeWrapper.addEventListener("mouseleave", () => {
    loupeLens.style.display = "none";
  });

  loupeWrapper.addEventListener("touchstart", (e) => {
    if (e.touches.length > 0) {
      updateLoupe(e.touches[0].clientX, e.touches[0].clientY);
    }
  }, { passive: true });

  loupeWrapper.addEventListener("touchmove", (e) => {
    if (e.touches.length > 0) {
      updateLoupe(e.touches[0].clientX, e.touches[0].clientY);
    }
  }, { passive: true });

  loupeWrapper.addEventListener("touchend", () => {
    loupeLens.style.display = "none";
  });

  // ==========================================================================
  // DRAG & DROP AND FILE SELECTION
  // ==========================================================================
  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  previewClose.addEventListener("click", () => {
    currentFile = null;
    currentImageSrc = "";
    fileInput.value = "";
    previewContainer.style.display = "none";
    dropzone.style.display = "block";
    btnAnalyze.disabled = true;
    loupeLens.style.display = "none";
  });

  function handleFileSelected(file) {
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      currentImageSrc = e.target.result;
      previewImage.src = currentImageSrc;
      previewContainer.style.display = "block";
      dropzone.style.display = "none";
      btnAnalyze.disabled = false;
      applyDarkroomFilters();
    };
    reader.readAsDataURL(file);
  }

  // Preset Buttons (quick test on standard cases)
  const presetBtns = document.querySelectorAll(".preset-btn");
  presetBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const sampleId = btn.getAttribute("data-sample");
      loadShowcaseSample(sampleId);
    });
  });

  // ==========================================================================
  // LIVE INFERENCE EXECUTION
  // ==========================================================================
  btnAnalyze.addEventListener("click", async () => {
    if (!currentFile) return;

    setLoading(true);
    const formData = new FormData();
    formData.append("file", currentFile);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Prediction request failed");
      }

      const data = await response.json();
      currentLastResult = data;
      renderInferenceResults(data);

      // Save to Session History
      saveToHistory({
        id: data.filename || "Uploaded Scan",
        timestamp: new Date().toLocaleTimeString(),
        image_src: currentImageSrc,
        lora_class: data.lora_prediction.display_name,
        lora_conf: data.lora_prediction.confidence_pct,
        frozen_class: data.frozen_prediction.display_name,
        frozen_conf: data.frozen_prediction.confidence_pct,
        is_tumor: data.lora_prediction.is_tumor,
        is_rescued: data.comparison.is_rescued_by_lora,
        latency_ms: data.lora_prediction.latency_ms
      });

      resultsContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (err) {
      alert("Inference Error: " + err.message);
    } finally {
      setLoading(false);
    }
  });

  function setLoading(isLoading) {
    if (isLoading) {
      btnAnalyze.disabled = true;
      analyzeSpinner.style.display = "inline-block";
      analyzeBtnText.textContent = "Analyzing Scan...";
    } else {
      btnAnalyze.disabled = false;
      analyzeSpinner.style.display = "none";
      analyzeBtnText.textContent = "Run Diagnostic Inference →";
    }
  }

  function renderInferenceResults(data) {
    resultsContainer.style.display = "block";
    emptyState.style.display = "none";

    const lora = data.lora_prediction;
    const frozen = data.frozen_prediction;
    const comp = data.comparison;

    // 1. LoRA Model Card
    loraClass.textContent = lora.display_name;
    loraClass.style.color = lora.color;
    loraPathology.textContent = lora.pathology_tag;
    loraPathology.className = `pathology-tag ${lora.is_tumor ? "tag-tumor" : "tag-healthy"}`;
    loraConfidence.textContent = `Confidence: ${lora.confidence_pct}%`;
    loraLatency.textContent = `${lora.latency_ms} ms`;
    renderProbBars(loraProbBars, lora.probabilities);

    // 2. Frozen Baseline Card
    frozenClass.textContent = frozen.display_name;
    frozenClass.style.color = frozen.color;
    frozenPathology.textContent = frozen.pathology_tag;
    frozenPathology.className = `pathology-tag ${frozen.is_tumor ? "tag-tumor" : "tag-healthy"}`;
    frozenConfidence.textContent = `Confidence: ${frozen.confidence_pct}%`;
    frozenLatency.textContent = `${frozen.latency_ms} ms`;
    renderProbBars(frozenProbBars, frozen.probabilities);

    // 3. LoRA Rescued Callout
    if (comp && comp.is_rescued_by_lora) {
      rescuedAlert.style.display = "flex";
      rescuedAlert.innerHTML = `
        <span style="font-size: 20px;">✓</span>
        <div>
          <strong>LoRA Rescued Baseline Error:</strong> The frozen baseline misclassified this scan as 
          <em>${frozen.display_name}</em>. QKV-LoRA representation adaptation successfully resolved this ambiguity and correctly diagnosed 
          <strong>${lora.display_name}</strong>.
        </div>
      `;
    } else {
      rescuedAlert.style.display = "none";
    }
  }

  function renderProbBars(container, probs) {
    container.innerHTML = "";
    Object.keys(probs).forEach((cls) => {
      const pct = probs[cls];
      const color = CLASS_COLORS[cls] || "#9B43F3";
      const label = CLASS_LABELS[cls] || cls;

      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <div class="prob-meta">
          <span style="color: ${color}; font-weight: 500;">${label}</span>
          <span style="font-family: var(--font-family-mono); font-weight: 600;">${pct.toFixed(2)}%</span>
        </div>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width: ${Math.max(pct, 1)}%; background: ${color};"></div>
        </div>
      `;
      container.appendChild(row);
    });
  }

  // ==========================================================================
  // SESSION HISTORY MANAGER (LocalStorage Persistence & Export)
  // ==========================================================================
  function getHistory() {
    try {
      const stored = localStorage.getItem("neurodino_history");
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  }

  function saveToHistory(item) {
    const list = getHistory();
    list.unshift(item);
    if (list.length > 50) list.pop(); // Keep last 50 scans
    localStorage.setItem("neurodino_history", JSON.stringify(list));
    renderHistoryTable();
  }

  function renderHistoryTable() {
    const list = getHistory();
    historyCounter.textContent = list.length;
    historyTableBody.innerHTML = "";

    if (list.length === 0) {
      historyTableBody.innerHTML = `
        <tr>
          <td colspan="7" style="text-align: center; padding: 40px; color: var(--color-mono-70);">
            No scans analyzed yet in this session. Run an analysis above to populate this log.
          </td>
        </tr>
      `;
      return;
    }

    list.forEach((item, idx) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><img class="history-thumb" src="${item.image_src}" alt="Thumb" /></td>
        <td style="font-family: var(--font-family-mono); font-size: 12px;">${item.timestamp}</td>
        <td><strong style="color: #FFF;">${item.id}</strong></td>
        <td>
          <span style="color: var(--color-primary); font-weight: 600;">${item.lora_class}</span>
          <span style="font-size: 12px; color: var(--color-mono-70);">(${item.lora_conf}%)</span>
        </td>
        <td>
          <span>${item.frozen_class}</span>
          <span style="font-size: 12px; color: var(--color-mono-70);">(${item.frozen_conf}%)</span>
        </td>
        <td>
          ${item.is_rescued ? '<span style="color: var(--color-success); font-weight: 700;">✓ Rescued</span>' : (item.lora_class === item.frozen_class ? '<span style="color: var(--color-mono-70);">Agreement</span>' : '<span style="color: var(--color-warning);">Discrepancy</span>')}
        </td>
        <td style="font-family: var(--font-family-mono); font-size: 12px;">${item.latency_ms} ms</td>
      `;
      historyTableBody.appendChild(tr);
    });
  }

  btnClearHistory.addEventListener("click", () => {
    if (confirm("Clear all scans from current session history?")) {
      localStorage.removeItem("neurodino_history");
      renderHistoryTable();
    }
  });

  btnExportCsv.addEventListener("click", () => {
    const list = getHistory();
    if (list.length === 0) {
      alert("No session history to export.");
      return;
    }

    let csvContent = "data:text/csv;charset=utf-8,";
    csvContent += "Timestamp,Sample_ID,LoRA_Diagnosis,LoRA_Confidence_Pct,Frozen_Diagnosis,Frozen_Confidence_Pct,Rescued_Status,Latency_ms\n";

    list.forEach((row) => {
      csvContent += `"${row.timestamp}","${row.id}","${row.lora_class}",${row.lora_conf},"${row.frozen_class}",${row.frozen_conf},"${row.is_rescued ? 'RESCUED' : 'MATCH'}",${row.latency_ms}\n`;
    });

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `neurodino_session_history_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });

  btnPrintReport.addEventListener("click", () => {
    window.print();
  });

  // ==========================================================================
  // LATENT REPRESENTATION & PCA/UMAP RESULTS
  // ==========================================================================
  const PROJECTION_VIEWS = {
    pca: {
      title: "Figure 2A: 2D Principal Component Analysis (PCA)",
      desc: "Linear projection of high-dimensional ViT CLS embeddings capturing global representation variance across 1,357 test scans. DINOv2 with QKV-LoRA establishes distinct linear separation between all 4 classes.",
      img: "/evidence-assets/verified_figures/fig2_pca_2d_projection.png"
    },
    umap: {
      title: "Figure 2B: 2D Uniform Manifold Approximation & Projection (UMAP)",
      desc: "Non-linear topological manifold projection highlighting local neighborhood clustering purity (93.93%). Notice the sharp separation boundary achieved between Glioma and Meningioma.",
      img: "/evidence-assets/verified_figures/fig2_umap_2d_projection.png"
    },
    confusion: {
      title: "Figure 1: Quad-Panel Confusion Matrix Evolution",
      desc: "Comprehensive breakdown showing Frozen Baseline (Arm A, 1,250 correct), Controlled LoRA (Arm B, 1,292 correct), and the Difference Matrix proving 47 net errors resolved by representation adaptation.",
      img: "/evidence-assets/verified_figures/fig1_confusion_matrices.png"
    },
    ceiling: {
      title: "Figure 3: Representation Ceiling Analysis",
      desc: "Linear probing across 7 classical downstream classifiers proved frozen foundation features hit an unyielding 92.11% accuracy ceiling, which only attention-level QKV adaptation could shatter.",
      img: "/evidence-assets/verified_figures/fig3_representation_ceiling.png"
    }
  };

  resultsTabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      resultsTabBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      const view = btn.getAttribute("data-view");
      const config = PROJECTION_VIEWS[view];
      if (config) {
        projectionImg.src = config.img;
        projectionTitle.textContent = config.title;
        projectionDesc.textContent = config.desc;
      }
    });
  });

  async function loadRepresentationMetrics() {
    try {
      const res = await fetch("/api/representation-metrics");
      const data = await res.json();
      geomTableBody.innerHTML = "";

      data.geometry.forEach((row) => {
        const tr = document.createElement("tr");
        const deltaVal = parseFloat(row.delta);
        const deltaFormatted = isNaN(deltaVal) ? row.delta : (deltaVal > 0 ? `+${deltaVal.toFixed(4)}` : deltaVal.toFixed(4));
        const deltaColor = (deltaVal > 0 && !row.metric.includes("Davies-Bouldin")) || (deltaVal < 0 && row.metric.includes("Davies-Bouldin")) ? "var(--color-success)" : "#FFF";

        tr.innerHTML = `
          <td><strong>${row.metric}</strong></td>
          <td style="font-family: var(--font-family-mono);">${parseFloat(row.frozen_control).toFixed(4)}</td>
          <td style="font-family: var(--font-family-mono); color: var(--color-primary); font-weight: 600;">${parseFloat(row.controlled_lora).toFixed(4)}</td>
          <td style="font-family: var(--font-family-mono); color: ${deltaColor}; font-weight: 700;">${deltaFormatted}</td>
        `;
        geomTableBody.appendChild(tr);
      });
    } catch (err) {
      console.error("Failed to load representation metrics:", err);
    }
  }

  // ==========================================================================
  // RESEARCH GENESIS STORYTELLING TIMELINE
  // ==========================================================================
  async function loadStoryMilestones() {
    try {
      const res = await fetch("/api/story-milestones");
      const data = await res.json();
      storyTimelineContainer.innerHTML = "";

      data.milestones.forEach((m) => {
        const item = document.createElement("div");
        item.className = "story-item";

        let chipsHtml = "";
        Object.keys(m.stats).forEach((k) => {
          chipsHtml += `<div class="story-chip">${k}: <strong>${m.stats[k]}</strong></div>`;
        });

        item.innerHTML = `
          <div class="story-circle">${m.step}</div>
          <div class="story-meta">
            <span class="story-badge">${m.badge}</span>
            <span class="story-date">${m.date}</span>
          </div>
          <h3 class="story-title">${m.title}</h3>
          <p class="story-desc">${m.summary}</p>
          <div class="story-image-box">
            <img class="story-img" src="${m.image}" alt="${m.title}" loading="lazy" />
          </div>
          <div class="story-stat-chips">${chipsHtml}</div>
        `;
        storyTimelineContainer.appendChild(item);
      });
    } catch (err) {
      console.error("Failed to load story milestones:", err);
    }
  }

  // ==========================================================================
  // SHOWCASE 50 GALLERY LOADER
  // ==========================================================================
  async function loadShowcaseManifest() {
    try {
      const response = await fetch("/api/showcase");
      const data = await response.json();
      allShowcaseScans = data.scans;
      renderShowcaseGrid(allShowcaseScans);
    } catch (err) {
      console.error("Failed to load showcase manifest:", err);
    }
  }

  function renderShowcaseGrid(scans) {
    showcaseGrid.innerHTML = "";
    scans.forEach((scan) => {
      const card = document.createElement("div");
      card.className = "showcase-card";
      const color = CLASS_COLORS[scan.true_class] || "#9B43F3";

      card.innerHTML = `
        <img class="showcase-card-thumb" src="${scan.image_url}" alt="${scan.sample_id}" loading="lazy" />
        <div class="showcase-card-header">
          <span class="showcase-card-title">${scan.sample_id}</span>
          <span class="showcase-badge" style="background: ${color}20; color: ${color}; border: 1px solid ${color}40;">
            ${scan.true_class.toUpperCase()}
          </span>
        </div>
        <div style="font-size: 12px; color: var(--color-mono-70); margin-top: auto;">
          LoRA Conf: <strong style="color: #FFF;">${(scan.confidence * 100).toFixed(1)}%</strong>
        </div>
      `;

      card.addEventListener("click", () => {
        loadShowcaseSample(scan.sample_id);
      });

      showcaseGrid.appendChild(card);
    });
  }

  filterPills.forEach((pill) => {
    pill.addEventListener("click", () => {
      filterPills.forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");

      const category = pill.getAttribute("data-filter");
      if (category === "all") {
        renderShowcaseGrid(allShowcaseScans);
      } else {
        const filtered = allShowcaseScans.filter((s) => s.true_class === category);
        renderShowcaseGrid(filtered);
      }
    });
  });

  async function loadShowcaseSample(sampleId) {
    try {
      setLoading(true);
      const res = await fetch(`/api/showcase/predict/${sampleId}`, { method: "POST" });
      if (!res.ok) throw new Error("Could not predict showcase scan");
      const data = await res.json();

      currentImageSrc = data.recorded_showcase.image_url;
      previewImage.src = currentImageSrc;
      previewContainer.style.display = "block";
      dropzone.style.display = "none";
      btnAnalyze.disabled = false;
      applyDarkroomFilters();

      renderInferenceResults(data);

      saveToHistory({
        id: sampleId,
        timestamp: new Date().toLocaleTimeString(),
        image_src: currentImageSrc,
        lora_class: data.lora_prediction.display_name,
        lora_conf: data.lora_prediction.confidence_pct,
        frozen_class: data.frozen_prediction.display_name,
        frozen_conf: data.frozen_prediction.confidence_pct,
        is_tumor: data.lora_prediction.is_tumor,
        is_rescued: data.comparison.is_rescued_by_lora,
        latency_ms: data.lora_prediction.latency_ms
      });

      resultsContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (err) {
      alert("Error predicting showcase sample: " + err.message);
    } finally {
      setLoading(false);
    }
  }

  // ==========================================================================
  // ACCORDIONS (Matching user Image 3 & PDF)
  // ==========================================================================
  const accordionItems = document.querySelectorAll(".accordion-item");
  accordionItems.forEach((item) => {
    const header = item.querySelector(".accordion-header");
    const icon = item.querySelector(".accordion-icon");
    header.addEventListener("click", () => {
      const isOpen = item.classList.contains("open");
      accordionItems.forEach((other) => {
        other.classList.remove("open");
        const otherIcon = other.querySelector(".accordion-icon");
        if (otherIcon) otherIcon.textContent = "+";
      });
      if (!isOpen) {
        item.classList.add("open");
        if (icon) icon.textContent = "−";
      }
    });
  });

  if (accordionItems.length > 0) {
    accordionItems[0].classList.add("open");
    const firstIcon = accordionItems[0].querySelector(".accordion-icon");
    if (firstIcon) firstIcon.textContent = "−";
  }

  // Initial Boot
  loadShowcaseManifest();
  loadStoryMilestones();
  loadRepresentationMetrics();
  renderHistoryTable();
});
