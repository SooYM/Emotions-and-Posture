# 🔮 Client-Side Multi-Modal Emotion & Pose Analyzer

A high-performance, **100% client-side** web application that performs real-time facial emotion recognition and body posture tracking directly in the browser. Using local machine learning execution, the application guarantees total privacy and zero server overhead.

---

## 🚀 Key Features

*   **⚡ Multiple Inference Engines**:
    *   **🤖 Hugging Face ViT (Vision Transformer)**: Runs `mo-thecreator/vit-Facial-Expression-Recognition` locally in-browser using `@huggingface/transformers` (with support for WebGPU acceleration and WASM fallbacks).
    *   **🧠 Custom MLP Classifier**: Pure JavaScript implementation of a Multi-Layer Perceptron (MLP) neural network running on 52 face blendshape coordinates.
    *   **📐 Baseline Default (Neutral)**: Safe, flat-state fallback logic.
*   **📷 Live Camera Mode**: Real-time video frame processing at 30 FPS with interactive face and pose coordinate overlays.
*   **📁 Static Image Upload**: Drag-and-drop or browse local images to perform instant analysis.
*   **🧪 Dataset Validation Suite**:
    *   Evaluate model accuracy on custom or preloaded datasets (like FER2013).
    *   Generates a dynamic **Confusion Matrix** (Actual vs. Predicted grid).
    *   Per-emotion accuracy breakdown tables.
*   **📋 One-Click Logs Clipboard Copying**:
    *   One-click copies the live `Inference Event Feed` system logs.
    *   One-click formats and copies the entire validation scan output as a clean **CSV** string.
*   **🔒 Privacy First**: Fully local execution; no images or data ever leave the client browser.

---

## 🛠️ Technology Stack

1.  **Core Framework**: Vanilla HTML5, CSS3 Custom Properties (sleek dark mode design), and ES6 JavaScript.
2.  **Bundler & Dev Server**: [Vite 8](https://vite.dev) (configured with custom COOP/COEP headers to support multi-threaded WASM execution).
3.  **Local Inference Runtimes**:
    *   `@mediapipe/tasks-vision` (Face & Pose Landmarkers running locally).
    *   `@huggingface/transformers` (Local ONNX Runtime Web execution).

---

## 📁 Project Structure

```text
├── public/
│   ├── vit_onnx/                 # Vision Transformer ONNX model, config, and preprocessors
│   ├── Dataset/                  # Validation evaluation image files
│   ├── dataset_manifest.json     # Preloaded dataset splits list
│   ├── model_weights.json        # Pre-trained MLP weights matrix
│   └── face_landmarker.task      # MediaPipe model assets
├── src/
│   ├── main.js                   # Application coordinator, camera stream loop, and validation suite
│   └── emotion.js                # Inference loading rules, MLP evaluation, and ViT integration
├── index.html                    # Dashboard UI layout
├── vite.config.js                # Vite server headers & dependency options configuration
├── package.json                  # Node dependencies & npm commands
└── README.md                     # Project documentation
```

---

## 🏁 Getting Started

### 1. Prerequisites
Install [Node.js](https://nodejs.org) (v18 or higher recommended).

### 2. Installation
Clone this repository to your workspace, navigate to the folder, and install dependencies:
```bash
npm install
```

### 3. Launch Development Server
Start the local development server:
```bash
npm run dev
```
Open the provided local URL (typically `http://localhost:5173`) in your web browser.

### 4. Build for Production
To package the app into a compressed, production-ready static bundle inside the `dist/` directory:
```bash
npm run build
```

---

## 💡 Troubleshooting & Implementation Details

### 1. "this.processor is not a function" (Cache Poisoning)
*   **Cause**: If the web page previously fetched a `404` error template when loading the model configurations, the browser's Cache API stored that invalid page. Refreshing the browser did not clear this custom cache layer, breaking the model processor initialization.
*   **Solution**: We configured `env.useBrowserCache = false` in `src/emotion.js` to disable the custom Cache API storage. Assets are read directly from disk and cached using standard HTTP headers instead.

### 2. High-Performance Multi-Threading (SharedArrayBuffer)
*   **Requirement**: MediaPipe and ONNX Runtime require WebAssembly multi-threading to achieve high framerates.
*   **Solution**: `vite.config.js` is programmed with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` headers to isolate the browser thread, enabling multi-threaded execution.

### 3. Tiny Dataset Images (MediaPipe Bypassing)
*   **Cause**: Evaluation datasets (like FER2013) use `48x48` pixel images. Google MediaPipe's Face Landmarker is designed for webcam streams and fails to identify faces on tiny dimensions.
*   **Solution**: The **Vision Transformer (ViT)** validation pipeline runs directly on the image canvas, bypassing MediaPipe landmark requirements. This lets you run validation on any low-resolution pre-cropped dataset files with true accuracy.
