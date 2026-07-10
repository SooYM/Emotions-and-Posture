import { detectorInstance } from "./detector.js";
import { classifyEmotion, EMOTION_METADATA, EMOTIONS, loadModelWeights, classifyEmotionViT, classifyEmotionCNN, selectedModelEngine, setSelectedModelEngine, vitPipeline, cnnSession, modelWeights, modelLoadErrors } from "./emotion.js";
import { classifyPosture, POSTURE_METADATA, POSTURES } from "./posture.js";
import { Visualizer } from "./visualizer.js";

// DOM Cache
const dom = {
  loadingOverlay: document.getElementById("loading-overlay"),
  loadingStatus: document.getElementById("loading-status"),
  loadingBar: document.getElementById("loading-bar"),
  modelStatusText: document.getElementById("model-status-text"),
  
  // Tabs
  tabBtnCamera: document.getElementById("tab-btn-camera"),
  tabBtnUpload: document.getElementById("tab-btn-upload"),
  tabContentCamera: document.getElementById("tab-content-camera"),
  tabContentUpload: document.getElementById("tab-content-upload"),
  
  // Camera Elements
  webcamVideo: document.getElementById("webcam-video"),
  webcamCanvas: document.getElementById("webcam-canvas"),
  webcamPlaceholder: document.getElementById("webcam-placeholder"),
  btnStartCamera: document.getElementById("btn-start-camera"),
  btnStopCamera: document.getElementById("btn-stop-camera"),
  valFps: document.getElementById("val-fps"),
  
  // Upload Elements
  uploadPreview: document.getElementById("upload-preview"),
  uploadCanvas: document.getElementById("upload-canvas"),
  dropZone: document.getElementById("drop-zone"),
  fileInput: document.getElementById("file-input"),
  btnBrowseFile: document.getElementById("btn-browse-file"),
  btnResetUpload: document.getElementById("btn-reset-upload"),
  uploadStatus: document.getElementById("upload-status"),
  
  // Settings
  chkShowFace: document.getElementById("chk-show-face"),
  chkShowPose: document.getElementById("chk-show-pose"),
  
  // Emotion Results
  emotionEmoji: document.getElementById("emotion-emoji"),
  emotionLabel: document.getElementById("emotion-label"),
  emotionConfidence: document.getElementById("emotion-confidence"),
  
  // Emotion Bars
  barHappy: document.getElementById("bar-happy"),
  valHappy: document.getElementById("val-happy"),
  barSad: document.getElementById("bar-sad"),
  valSad: document.getElementById("val-sad"),
  barAnxiety: document.getElementById("bar-anxiety"),
  valAnxiety: document.getElementById("val-anxiety"),
  barAngry: document.getElementById("bar-angry"),
  valAngry: document.getElementById("val-angry"),
  barSurprised: document.getElementById("bar-surprised"),
  valSurprised: document.getElementById("val-surprised"),
  barDisgust: document.getElementById("bar-disgust"),
  valDisgust: document.getElementById("val-disgust"),
  barNeutral: document.getElementById("bar-neutral"),
  valNeutral: document.getElementById("val-neutral"),
  
  // Posture Results
  postureIcon: document.getElementById("posture-icon"),
  postureLabel: document.getElementById("posture-label"),
  postureDetails: document.getElementById("posture-details"),
  valHipAngle: document.getElementById("val-hip-angle"),
  valKneeAngle: document.getElementById("val-knee-angle"),
  valTrunkAngle: document.getElementById("val-trunk-angle"),
  
  // Logs
  logsContainer: document.getElementById("logs-container"),

  // Validation Elements
  tabBtnValidation: document.getElementById("tab-btn-validation"),
  tabContentValidation: document.getElementById("tab-content-validation"),
  validationResults: document.getElementById("validation-results"),
  realtimeResults: document.getElementById("realtime-results"),
  validationDirInput: document.getElementById("validation-dir-input"),
  btnSelectValidationDir: document.getElementById("btn-select-validation-dir"),
  validationLimit: document.getElementById("validation-limit"),
  validationScanStatus: document.getElementById("validation-scan-status"),
  validationProgressPanel: document.getElementById("validation-progress-panel"),
  validationRunStatus: document.getElementById("validation-run-status"),
  validationSpeed: document.getElementById("validation-speed"),
  validationProgressFill: document.getElementById("validation-progress-fill"),
  valProcessedCount: document.getElementById("val-processed-count"),
  valTotalCount: document.getElementById("val-total-count"),
  btnRunValidation: document.getElementById("btn-run-validation"),
  btnPauseValidation: document.getElementById("btn-pause-validation"),
  btnCancelValidation: document.getElementById("btn-cancel-validation"),
  validationPreviewCanvas: document.getElementById("validation-preview-canvas"),
  valPreviewFilename: document.getElementById("val-preview-filename"),
  valPreviewActual: document.getElementById("val-preview-actual"),
  valPreviewPredict: document.getElementById("val-preview-predict"),
  valStatAccuracy: document.getElementById("val-stat-accuracy"),
  valStatProcessed: document.getElementById("val-stat-processed"),
  valStatDetectionRate: document.getElementById("val-stat-detection-rate"),
  valBreakdownTbody: document.getElementById("val-breakdown-tbody"),
  confusionMatrixGrid: document.getElementById("confusion-matrix-grid"),
  valLogsTbody: document.getElementById("val-logs-tbody"),
  radioSourcePreloaded: document.getElementById("radio-source-preloaded"),
  radioSourceCustom: document.getElementById("radio-source-custom"),
  valPreloadedCount: document.getElementById("val-preloaded-count"),
  customFolderActions: document.getElementById("custom-folder-actions"),
  validationSplitFilter: document.getElementById("validation-split-filter"),
  selectModelEngine: document.getElementById("select-model-engine"),
  activeModelLabel: document.getElementById("active-model-label"),
  activeModelDot: document.getElementById("active-model-dot"),
  validationModelEngine: document.getElementById("validation-model-engine"),
  btnCopySystemLogs: document.getElementById("btn-copy-system-logs"),
  btnCopyValLogs: document.getElementById("btn-copy-val-logs")
};

// Global App State
let activeTab = "camera"; // "camera", "upload", or "validation"
let cameraStream = null;
let isCameraRunning = false;
let animationFrameId = null;
let lastFrameTime = 0;
let fpsList = [];
let cachedLastResults = { faceResult: null, poseResult: null };
let isEmotionInferenceRunning = false;
let lastEmotionResult = null;

// Validation Suite State
let validationQueue = [];
let preloadedManifest = []; // Parsed list from public/dataset_manifest.json
let selectedDatasetSource = "preloaded"; // "preloaded" or "custom"

let isValidationRunning = false;
let validationIndex = 0;
let validationLogsHistory = [];
let validationProcessedCount = 0;
let validationSuccessCount = 0;
let validationFacesDetected = 0;
let validationStartTime = 0;
let validationTotalTime = 0;
let validationResultsData = []; // Array of { file, trueLabel, predictedLabel, confidence, match, speed }

// Confusion matrix structure (mapping Actual -> Predicted)
// Emotions list ordered:
const VALIDATION_EMOTIONS = [
  EMOTIONS.HAPPY,
  EMOTIONS.SAD,
  EMOTIONS.ANXIETY,
  EMOTIONS.ANGRY,
  EMOTIONS.SURPRISED,
  EMOTIONS.DISGUST,
  EMOTIONS.NEUTRAL
];
let confusionMatrix = {};
function resetConfusionMatrix() {
  confusionMatrix = {};
  for (const act of VALIDATION_EMOTIONS) {
    confusionMatrix[act] = {};
    for (const pred of VALIDATION_EMOTIONS) {
      confusionMatrix[act][pred] = 0;
    }
  }
}
resetConfusionMatrix();

// --- Console Redirector for Telemetry and UI Diagnostics ---
const originalConsoleError = console.error;
console.error = function(...args) {
  originalConsoleError.apply(console, args);
  const msg = args.map(a => {
    if (a instanceof Error) return a.message;
    if (typeof a === 'object') {
      try { return JSON.stringify(a); } catch { return String(a); }
    }
    return String(a);
  }).join(' ');
  if (typeof dom !== 'undefined' && dom.logsContainer) {
    addLog(`Console Error: ${msg}`, "error");
  }
};

const originalConsoleWarn = console.warn;
console.warn = function(...args) {
  originalConsoleWarn.apply(console, args);
  const msg = args.map(a => {
    if (a instanceof Error) return a.message;
    if (typeof a === 'object') {
      try { return JSON.stringify(a); } catch { return String(a); }
    }
    return String(a);
  }).join(' ');
  if (typeof dom !== 'undefined' && dom.logsContainer) {
    addLog(`Console Warning: ${msg}`, "warning");
  }
};

// --- Logger Helper ---
function addLog(text, type = "info") {
  const time = new Date().toLocaleTimeString();
  const entry = document.createElement("div");
  entry.className = `log-entry ${type}`;
  entry.innerText = `[${time}] ${text}`;
  dom.logsContainer.appendChild(entry);
  dom.logsContainer.scrollTop = dom.logsContainer.scrollHeight;
}

// --- App Initialization ---
async function initializeApp() {
  try {
    addLog("App initialization started.", "system");
    
    // Animate fake progress steps for beautiful loading visual
    let progress = 10;
    const progressInterval = setInterval(() => {
      if (progress < 90) {
        progress += Math.floor(Math.random() * 8) + 2;
        dom.loadingBar.style.width = `${progress}%`;
      }
    }, 120);

    await detectorInstance.init((statusMsg) => {
      dom.loadingStatus.innerText = statusMsg;
      addLog(statusMsg, "system");
    });

    // Start background parallel loading of MLP, CNN and ViT models
    // This allows the app overlay to clear and the webcam to stream instantly
    loadModelWeights((modelType) => {
      if (modelType === "mlp") {
        addLog("🧠 Custom Neural Net (MLP) weights loaded.", "success");
      } else if (modelType === "cnn") {
        addLog("🌀 Custom CNN (FERNet) model loaded successfully.", "success");
      } else if (modelType === "vit") {
        addLog("🤖 Hugging Face ViT model loaded successfully.", "success");
      }
      
      // Dynamically update UI active labels as models become available
      let currentActive = dom.selectModelEngine.value;
      if (currentActive === "auto") {
        currentActive = vitPipeline ? "vit" : (cnnSession ? "cnn" : (modelWeights ? "mlp" : "heuristics"));
      }
      updateActiveModelLabel(currentActive);
    }).then(() => {
      dom.modelStatusText.innerText = "Models Loaded: MediaPipe, Custom MLP, Custom CNN, and Hugging Face ViT";
      
      if (modelLoadErrors.mlp) {
        addLog(`MLP model failed to load: ${modelLoadErrors.mlp}`, "warning");
      }
      if (modelLoadErrors.vitCPU) {
        addLog(`ViT model failed to load: ${modelLoadErrors.vitCPU}`, "error");
      }
      if (modelLoadErrors.cnnCPU) {
        addLog(`CNN model failed to load: ${modelLoadErrors.cnnCPU}`, "error");
      }
    });

    clearInterval(progressInterval);
    dom.loadingBar.style.width = "100%";
    
    // Smooth transition out for loading screen
    setTimeout(() => {
      dom.loadingOverlay.classList.add("fade-out");
      setTimeout(() => {
        dom.loadingOverlay.style.display = "none";
      }, 500);
      addLog("System ready. WebAssembly models online.", "system");
      initializeDatasetValidation(); // Load default folder list on load
    }, 400);

  } catch (error) {
    console.error("System initialization failed:", error);
    dom.loadingStatus.innerHTML = `<span style="color:#EF4444">Initialization Failed: ${error.message}</span>`;
    addLog(`System initialization failed: ${error.message}`, "error");
    dom.modelStatusText.innerText = "Error: MediaPipe landmarker models failed to load. Camera tracking disabled.";
    dom.modelStatusText.style.color = "#EF4444";

    clearInterval(progressInterval);
    dom.loadingBar.style.width = "100%";
    dom.loadingBar.style.backgroundColor = "#EF4444";

    // Append a dismiss button to allow entering the app in fallback mode
    const existingBtn = dom.loadingOverlay.querySelector(".dismiss-btn");
    if (!existingBtn) {
      const dismissBtn = document.createElement("button");
      dismissBtn.className = "dismiss-btn";
      dismissBtn.innerText = "Dismiss and Enter App";
      dismissBtn.style.marginTop = "20px";
      dismissBtn.style.padding = "10px 20px";
      dismissBtn.style.backgroundColor = "#3B82F6";
      dismissBtn.style.color = "white";
      dismissBtn.style.border = "none";
      dismissBtn.style.borderRadius = "6px";
      dismissBtn.style.cursor = "pointer";
      dismissBtn.style.fontWeight = "bold";
      dismissBtn.onclick = () => {
        dom.loadingOverlay.classList.add("fade-out");
        setTimeout(() => {
          dom.loadingOverlay.style.display = "none";
        }, 500);
      };
      
      const container = dom.loadingOverlay.querySelector(".loading-card") || dom.loadingOverlay;
      container.appendChild(dismissBtn);
    }
  }
}

// --- Tab Management ---
function switchTab(tabName) {
  if (activeTab === tabName) return;
  activeTab = tabName;

  // Reset tab button states
  dom.tabBtnCamera.classList.toggle("active", activeTab === "camera");
  dom.tabBtnUpload.classList.toggle("active", activeTab === "upload");
  dom.tabBtnValidation.classList.toggle("active", activeTab === "validation");

  // Reset tab content panes
  dom.tabContentCamera.style.display = activeTab === "camera" ? "block" : "none";
  dom.tabContentUpload.style.display = activeTab === "upload" ? "block" : "none";
  dom.tabContentValidation.style.display = activeTab === "validation" ? "block" : "none";

  // Toggle Results columns
  dom.realtimeResults.style.display = activeTab !== "validation" ? "block" : "none";
  dom.validationResults.style.display = activeTab === "validation" ? "block" : "none";

  // Handle stream controls and resets
  if (activeTab === "camera") {
    resetUploadUI();
    addLog("Switched to Live Camera mode.", "system");
  } else if (activeTab === "upload") {
    stopCameraStream();
    resetUploadUI();
    addLog("Switched to Image Upload mode.", "system");
  } else if (activeTab === "validation") {
    stopCameraStream();
    resetUploadUI();
    addLog("Switched to Dataset Validation mode.", "system");
    // Render initial empty validation statistics layout
    renderConfusionMatrix();
    renderValidationTable();
  }
  
  resetAnalysisUI();
}

// --- UI Reset Helpers ---
function resetAnalysisUI() {
  cachedLastResults = { faceResult: null, poseResult: null };
  
  // Reset emotion display
  dom.emotionEmoji.innerText = "😐";
  dom.emotionLabel.innerText = "Neutral";
  dom.emotionConfidence.innerText = "100%";
  dom.emotionLabel.style.color = EMOTION_METADATA[EMOTIONS.NEUTRAL].color;

  const barFills = ["happy", "sad", "anxiety", "angry", "surprised", "disgust"];
  barFills.forEach(emo => {
    dom[`bar${emo.charAt(0).toUpperCase() + emo.slice(1)}`].style.width = "0%";
    dom[`val${emo.charAt(0).toUpperCase() + emo.slice(1)}`].innerText = "0%";
  });
  dom.barNeutral.style.width = "100%";
  dom.valNeutral.innerText = "100%";

  // Reset posture display
  dom.postureIcon.innerText = "👤";
  dom.postureLabel.innerText = "Unknown";
  dom.postureLabel.style.color = POSTURE_METADATA[POSTURES.UNKNOWN].color;
  dom.postureDetails.innerText = "Awaiting full body detection. Hips and knees must be in view.";
  dom.valHipAngle.innerText = "--°";
  dom.valKneeAngle.innerText = "--°";
  dom.valTrunkAngle.innerText = "--°";
}

// --- Webcam Mode Controller ---
async function startCameraStream() {
  try {
    addLog("Requesting camera access...", "system");
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user"
      },
      audio: false
    });

    dom.webcamVideo.srcObject = cameraStream;
    
    dom.webcamVideo.onloadedmetadata = async () => {
      // Setup canvas size matching the metadata size of the video feed
      const videoWidth = dom.webcamVideo.videoWidth;
      const videoHeight = dom.webcamVideo.videoHeight;
      dom.webcamCanvas.width = videoWidth;
      dom.webcamCanvas.height = videoHeight;
      
      // Setup dynamic aspect ratio matching
      const aspect = videoWidth / videoHeight;
      dom.webcamVideo.parentElement.style.aspectRatio = `${aspect}`;
      
      dom.webcamVideo.play();
      
      // Update UI Controls
      dom.webcamPlaceholder.style.display = "none";
      dom.btnStartCamera.disabled = true;
      dom.btnStopCamera.disabled = false;
      isCameraRunning = true;
      
      addLog(`Webcam started: ${videoWidth}x${videoHeight}`, "system");
      
      // Ensure detector is running in VIDEO mode before entering the frame processing loop
      try {
        await detectorInstance.setRunningMode("VIDEO");
      } catch (err) {
        console.error("Failed to set MediaPipe running mode to VIDEO:", err);
      }
      
      // Start Detection Loop
      lastFrameTime = performance.now();
      fpsList = [];
      animationFrameId = requestAnimationFrame(processCameraFrame);
    };

  } catch (error) {
    console.error("Camera access denied or error:", error);
    addLog(`Camera Error: ${error.message}. Please check permissions.`, "error");
    alert(`Could not access webcam: ${error.message}\n\nPlease check camera permissions.`);
  }
}

function stopCameraStream() {
  if (!isCameraRunning) return;
  
  isCameraRunning = false;
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
  
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
  }
  
  dom.webcamVideo.srcObject = null;
  dom.webcamVideo.parentElement.style.aspectRatio = "4 / 3"; // Reset default aspect ratio
  dom.webcamPlaceholder.style.display = "flex";
  dom.btnStartCamera.disabled = false;
  dom.btnStopCamera.disabled = true;
  dom.valFps.innerText = "0";
  
  // Clear canvas
  const ctx = dom.webcamCanvas.getContext("2d");
  ctx.clearRect(0, 0, dom.webcamCanvas.width, dom.webcamCanvas.height);
  
  addLog("Camera stream stopped.", "system");
}

async function processCameraFrame() {
  if (!isCameraRunning) return;

  try {
    const timestamp = performance.now();
    
    // Calculate FPS
    const elapsed = timestamp - lastFrameTime;
    lastFrameTime = timestamp;
    const currentFps = Math.round(1000 / elapsed);
    fpsList.push(currentFps);
    if (fpsList.length > 30) fpsList.shift();
    const averageFps = Math.round(fpsList.reduce((a, b) => a + b, 0) / fpsList.length);
    dom.valFps.innerText = averageFps;

    // Run MediaPipe Vision task inference
    const results = detectorInstance.detectVideoFrame(dom.webcamVideo, timestamp);
    
    // Save results to cache for re-renders on checkbox toggle
    cachedLastResults = results;

    // Classify and display results
    updateAnalysisResults(results, dom.webcamVideo);

    // Draw overlays on canvas
    drawVisualOverlays(dom.webcamCanvas, results);

  } catch (err) {
    console.error("Error processing video frame:", err);
    addLog(`Frame processing error: ${err.message}`, "error");
  }

  // Continue webcam processing loop
  animationFrameId = requestAnimationFrame(processCameraFrame);
}

// --- Image Upload Mode Controller ---
function triggerFileBrowser() {
  dom.fileInput.click();
}

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (file) {
    processImageFile(file);
  }
}

function processImageFile(file) {
  if (!file.type.startsWith("image/")) {
    addLog(`Rejected file: ${file.name} is not an image.`, "error");
    alert("Please select a valid image file.");
    return;
  }

  addLog(`Selected image: ${file.name} (${Math.round(file.size / 1024)} KB)`, "system");
  dom.uploadStatus.innerText = file.name;
  dom.btnResetUpload.disabled = false;

  const reader = new FileReader();
  reader.onload = (e) => {
    // Bind onload BEFORE setting src to prevent race conditions
    dom.uploadPreview.onload = async () => {
      // Clear onload callback to prevent multiple triggers
      dom.uploadPreview.onload = null;

      try {
        // Wait for modern browser image decoding to fully complete.
        // This ensures WebP, compressed formats, and large files are fully decoded 
        // in memory so that naturalWidth/naturalHeight are populated and readable by WebGL/MediaPipe.
        await dom.uploadPreview.decode();

        dom.dropZone.style.display = "none";
        dom.uploadPreview.style.display = "block";
        
        // Setup canvas size
        const naturalWidth = dom.uploadPreview.naturalWidth;
        const naturalHeight = dom.uploadPreview.naturalHeight;

        if (naturalWidth === 0 || naturalHeight === 0) {
          throw new Error("Loaded image has 0 width or height.");
        }

        dom.uploadCanvas.width = naturalWidth;
        dom.uploadCanvas.height = naturalHeight;

        // Draw image onto canvas to force browser rasterization and prevent WebP texture load lag
        const ctx = dom.uploadCanvas.getContext("2d");
        ctx.clearRect(0, 0, naturalWidth, naturalHeight);
        ctx.drawImage(dom.uploadPreview, 0, 0, naturalWidth, naturalHeight);

        // Setup dynamic aspect ratio matching
        const aspect = naturalWidth / naturalHeight;
        dom.uploadPreview.parentElement.style.aspectRatio = `${aspect}`;
        
        addLog(`Image loaded & rasterized (${naturalWidth}x${naturalHeight}). Running inferences...`, "system");
        
        // Run MediaPipe inference on the fully rasterized canvas element
        const results = await detectorInstance.detectImage(dom.uploadCanvas);
        cachedLastResults = results;

        // Clear canvas so it is transparent and only overlays are drawn on top of the preview image
        ctx.clearRect(0, 0, naturalWidth, naturalHeight);
        
        // Update dashboard
        updateAnalysisResults(results, dom.uploadPreview);
        
        // Draw overlays
        drawVisualOverlays(dom.uploadCanvas, results);
        
        addLog(`Inference completed. Emotion & posture detected.`, "detect");
      } catch (err) {
        console.error("Error decoding or analyzing uploaded image:", err);
        addLog(`Image inference error: ${err.message}`, "error");
        alert("Failed to analyze image: " + err.message);
      }
    };
    // Set src second
    dom.uploadPreview.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

function resetUploadUI() {
  dom.uploadPreview.style.display = "none";
  dom.uploadPreview.onload = null; // Clear handler
  dom.uploadPreview.src = "";
  dom.uploadPreview.parentElement.style.aspectRatio = "4 / 3"; // Reset default aspect ratio
  dom.dropZone.style.display = "flex";
  dom.fileInput.value = "";
  dom.btnResetUpload.disabled = true;
  dom.uploadStatus.innerText = "No file selected";
  
  // Clear canvas
  const ctx = dom.uploadCanvas.getContext("2d");
  ctx.clearRect(0, 0, dom.uploadCanvas.width, dom.uploadCanvas.height);
}

// --- Drag & Drop ---
function handleDragOver(e) {
  e.preventDefault();
  dom.dropZone.classList.add("dragover");
}

function handleDragLeave(e) {
  e.preventDefault();
  dom.dropZone.classList.remove("dragover");
}

function handleDrop(e) {
  e.preventDefault();
  dom.dropZone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) {
    processImageFile(file);
  }
}

// --- Classification Integration ---
function updateAnalysisResults(results, sourceElement = null) {
  const { faceResult, poseResult } = results;

  // Sort detected faces so that the largest face (closest to camera) is at index 0
  if (faceResult && faceResult.faceLandmarks && faceResult.faceLandmarks.length > 1) {
    const faceAreas = faceResult.faceLandmarks.map((landmarks, index) => {
      let minX = 1, maxX = 0, minY = 1, maxY = 0;
      for (const lm of landmarks) {
        if (lm.x < minX) minX = lm.x;
        if (lm.x > maxX) maxX = lm.x;
        if (lm.y < minY) minY = lm.y;
        if (lm.y > maxY) maxY = lm.y;
      }
      const area = (maxX - minX) * (maxY - minY);
      return { index, area };
    });

    // Sort descending
    faceAreas.sort((a, b) => b.area - a.area);

    // Reorder arrays
    const sortedLandmarks = faceAreas.map(item => faceResult.faceLandmarks[item.index]);
    const sortedBlendshapes = faceAreas.map(item => faceResult.faceBlendshapes[item.index]);

    faceResult.faceLandmarks = sortedLandmarks;
    faceResult.faceBlendshapes = sortedBlendshapes;
  }

  // 1. Trigger background async (ViT or CNN) emotion classification if camera frame or uploaded image is active
  const isVitActive = (selectedModelEngine === "vit" || (selectedModelEngine === "auto" && vitPipeline && !cnnSession)) && vitPipeline;
  const isCnnActive = (selectedModelEngine === "cnn" || (selectedModelEngine === "auto" && cnnSession)) && cnnSession;
  const isAsyncActive = isVitActive || isCnnActive;

  if (isAsyncActive && sourceElement && faceResult && faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0) {
    if (!isEmotionInferenceRunning) {
      isEmotionInferenceRunning = true;
      (async () => {
        try {
          const landmarks = faceResult.faceLandmarks[0];
          
          let srcW = sourceElement.videoWidth || sourceElement.naturalWidth || sourceElement.width;
          let srcH = sourceElement.videoHeight || sourceElement.naturalHeight || sourceElement.height;
          
          if (srcW && srcH) {
            // Get bounding box of the face landmarks
            let minX = 1, maxX = 0, minY = 1, maxY = 0;
            for (const lm of landmarks) {
              if (lm.x < minX) minX = lm.x;
              if (lm.x > maxX) maxX = lm.x;
              if (lm.y < minY) minY = lm.y;
              if (lm.y > maxY) maxY = lm.y;
            }
            
            // Add padding around face box
            const padX = (maxX - minX) * 0.15;
            const padY = (maxY - minY) * 0.15;
            minX = Math.max(0, minX - padX);
            maxX = Math.min(1, maxX + padX);
            minY = Math.max(0, minY - padY);
            maxY = Math.min(1, maxY + padY);
            
            const x = minX * srcW;
            const y = minY * srcH;
            const w = (maxX - minX) * srcW;
            const h = (maxY - minY) * srcH;
            
            if (w > 0 && h > 0) {
              const faceCanvas = document.createElement("canvas");
              const size = isVitActive ? 224 : 48; // ViT size is 224x224, CNN size is 48x48
              faceCanvas.width = size;
              faceCanvas.height = size;
              const ctx = faceCanvas.getContext("2d");
              ctx.drawImage(sourceElement, x, y, w, h, 0, 0, size, size);
              
              let asyncRes = null;
              if (isVitActive) {
                asyncRes = await classifyEmotionViT(faceCanvas);
              } else if (isCnnActive) {
                asyncRes = await classifyEmotionCNN(faceCanvas);
              }

              if (asyncRes && isAsyncActive) { // Ensure model selection didn't change mid-inference
                lastEmotionResult = asyncRes;
                updateEmotionUI(asyncRes);
              }
            }
          }
        } catch (e) {
          console.error("Background async classification failed:", e);
          addLog(`Async run fail: ${e.message}`, "error");
        } finally {
          isEmotionInferenceRunning = false;
        }
      })();
    }
  }

  // 1b. Emotion Classification - Fallback or active display
  let emotionRes = null;
  if (isAsyncActive) {
    if (faceResult && faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0) {
      emotionRes = lastEmotionResult;
    } else {
      lastEmotionResult = null; // Clear cached result if face tracking is lost
    }
  }

  // If no cached ViT result (or if running MLP/Baseline), evaluate synchronously
  if (!emotionRes) {
    emotionRes = classifyEmotion(faceResult, 0);
  }
  updateEmotionUI(emotionRes);

  // 2. Posture Classification
  const postureRes = classifyPosture(poseResult);
  const pMetadata = POSTURE_METADATA[postureRes.posture];

  dom.postureIcon.innerText = pMetadata.icon;
  dom.postureLabel.innerText = pMetadata.label;
  dom.postureLabel.style.color = pMetadata.color;
  dom.postureDetails.innerText = postureRes.details;

  // Update Joint Angle metrics
  const { hipAngle, kneeAngle, trunkAngle } = postureRes.angles;
  dom.valHipAngle.innerText = hipAngle !== null ? `${hipAngle}°` : "--°";
  dom.valKneeAngle.innerText = kneeAngle !== null ? `${kneeAngle}°` : "--°";
  dom.valTrunkAngle.innerText = trunkAngle !== null ? `${trunkAngle}°` : "--°";
}

function updateEmotionUI(emotionRes) {
  const metadata = EMOTION_METADATA[emotionRes.dominantEmotion];
  if (!metadata) return;
  
  dom.emotionEmoji.innerText = metadata.emoji;
  dom.emotionLabel.innerText = metadata.label;
  dom.emotionLabel.style.color = metadata.color;
  dom.emotionConfidence.innerText = `${Math.round(emotionRes.confidence * 100)}%`;

  // Update active model labels dynamically
  updateActiveModelLabel(emotionRes.activeModel || "heuristics");

  // Update bars in emotion breakdown
  for (const emotion in emotionRes.scores) {
    const rawVal = emotionRes.scores[emotion];
    const percentage = `${Math.round(rawVal * 100)}%`;
    const barElement = dom[`bar${emotion.charAt(0).toUpperCase() + emotion.slice(1)}`];
    const valElement = dom[`val${emotion.charAt(0).toUpperCase() + emotion.slice(1)}`];
    
    if (barElement) barElement.style.width = percentage;
    if (valElement) valElement.innerText = percentage;
  }
}

function updateActiveModelLabel(activeModel) {
  if (!dom.activeModelLabel || !dom.activeModelDot) return;
  
  const selected = dom.selectModelEngine.value;
  
  let labelText = "";
  let color = "";
  
  if (activeModel === "vit") {
    labelText = "🤖 Vision Transformer (ViT)";
    color = "#10B981"; // Emerald green
  } else if (activeModel === "cnn") {
    labelText = "🌀 Custom CNN (FERNet)";
    color = "#3B82F6"; // Blue
  } else if (activeModel === "mlp") {
    labelText = "🧠 Custom Neural Net (MLP)";
    color = "#8B5CF6"; // Purple
  } else {
    labelText = "📐 Baseline Default (Neutral)";
    color = "#F59E0B"; // Amber
  }
  
  // Highlight fallback warning in red
  if (selected !== "auto" && selected !== activeModel) {
    labelText += " (Fallback)";
    color = "#EF4444"; // Red
  }
  
  dom.activeModelLabel.innerText = labelText;
  dom.activeModelLabel.style.color = color;
  dom.activeModelDot.style.background = color;
}

// --- Drawing / Visualization Integration ---
function drawVisualOverlays(canvas, results) {
  const showFace = dom.chkShowFace.checked;
  const showPose = dom.chkShowPose.checked;
  
  const faceResult = showFace ? results.faceResult : null;
  const poseResult = showPose ? results.poseResult : null;

  const emotionRes = lastEmotionResult || classifyEmotion(results.faceResult);
  const metadata = EMOTION_METADATA[emotionRes.dominantEmotion];
  const emotionColor = metadata ? metadata.color : "#8B5CF6";

  const isMirrored = activeTab === "camera";
  Visualizer.draw(canvas, faceResult, poseResult, emotionColor, isMirrored);
}

function handleOverlayCheckboxChange() {
  // If we have cached results, re-draw them instantly when user toggles settings checkboxes
  const canvas = activeTab === "camera" ? dom.webcamCanvas : dom.uploadCanvas;
  if (cachedLastResults.faceResult || cachedLastResults.poseResult) {
    drawVisualOverlays(canvas, cachedLastResults);
  }
}

// --- Dataset Validation Suite Engine ---

function parseDatasetFile(file) {
  const path = file.webkitRelativePath || file.name;
  const parts = path.split('/');
  if (parts.length < 2) return null;
  
  // The category folder is the second to last element
  const category = parts[parts.length - 2].toLowerCase();
  
  // Translate dataset folder names to our internal emotion keys
  let trueLabel = null;
  if (category === "happy") trueLabel = EMOTIONS.HAPPY;
  else if (category === "sad") trueLabel = EMOTIONS.SAD;
  else if (category === "angry") trueLabel = EMOTIONS.ANGRY;
  else if (category === "disgust") trueLabel = EMOTIONS.DISGUST;
  else if (category === "surprise" || category === "surprised") trueLabel = EMOTIONS.SURPRISED;
  else if (category === "fear" || category === "anxiety") trueLabel = EMOTIONS.ANXIETY; // Map fear to anxiety
  else if (category === "neutral") trueLabel = EMOTIONS.NEUTRAL;
  
  if (!trueLabel) return null;
  
  // Detect train vs test split
  let split = "unknown";
  if (parts.length >= 3) {
    const parentDir = parts[parts.length - 3].toLowerCase();
    if (parentDir === "train" || parentDir === "test" || parentDir === "val" || parentDir === "validation") {
      split = parentDir;
    }
  }
  
  return {
    file,
    trueLabel,
    split,
    name: parts[parts.length - 1],
    relativePath: file.webkitRelativePath
  };
}

function loadImageFromFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("Failed to load image element"));
      img.src = e.target.result;
    };
    reader.onerror = () => reject(new Error("FileReader failed"));
    reader.readAsDataURL(file);
  });
}

function loadImageFromUrl(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Failed to load image URL: " + url));
    img.src = url;
  });
}

async function initializeDatasetValidation() {
  try {
    dom.validationScanStatus.className = "scan-status-alert info";
    dom.validationScanStatus.innerText = "Loading preloaded dataset manifest...";
    
    const response = await fetch("/dataset_manifest.json");
    if (!response.ok) {
      throw new Error(`Failed to load manifest: ${response.statusText}`);
    }
    const manifestPaths = await response.json();
    preloadedManifest = [];
    
    for (const path of manifestPaths) {
      const parts = path.split('/');
      const category = parts[parts.length - 2].toLowerCase();
      
      let trueLabel = null;
      if (category === "happy") trueLabel = EMOTIONS.HAPPY;
      else if (category === "sad") trueLabel = EMOTIONS.SAD;
      else if (category === "angry") trueLabel = EMOTIONS.ANGRY;
      else if (category === "disgust") trueLabel = EMOTIONS.DISGUST;
      else if (category === "surprise" || category === "surprised") trueLabel = EMOTIONS.SURPRISED;
      else if (category === "fear" || category === "anxiety") trueLabel = EMOTIONS.ANXIETY;
      else if (category === "neutral") trueLabel = EMOTIONS.NEUTRAL;
      
      if (!trueLabel) continue;
      
      let split = "unknown";
      if (parts.length >= 3) {
        const parentDir = parts[parts.length - 3].toLowerCase();
        if (parentDir === "train" || parentDir === "test" || parentDir === "val" || parentDir === "validation") {
          split = parentDir;
        }
      }
      
      preloadedManifest.push({
        relativePath: path,
        name: parts[parts.length - 1],
        trueLabel,
        split,
        isPreloaded: true
      });
    }
    
    dom.valPreloadedCount.innerText = preloadedManifest.length.toLocaleString();
    prepareDatasetQueue();
  } catch (err) {
    console.warn("Dataset manifest fetch failed. Custom folders will still work.", err);
    dom.validationScanStatus.className = "scan-status-alert error";
    dom.validationScanStatus.innerText = "Default dataset not found. Use Custom Folder option to run validation.";
    
    dom.radioSourcePreloaded.disabled = true;
    dom.radioSourceCustom.checked = true;
    toggleDatasetSource("custom");
  }
}

let customUploadedFiles = [];

function handleValidationDirSelect(event) {
  const files = Array.from(event.target.files);
  if (files.length === 0) return;
  
  customUploadedFiles = [];
  
  for (const file of files) {
    if (!/\.(jpe?g|png|webp)$/i.test(file.name)) continue;
    
    const item = parseDatasetFile(file);
    if (item) {
      customUploadedFiles.push(item);
    }
  }
  
  prepareDatasetQueue();
}

function toggleDatasetSource(source) {
  selectedDatasetSource = source;
  dom.customFolderActions.style.display = source === "custom" ? "block" : "none";
  prepareDatasetQueue();
}

function prepareDatasetQueue() {
  const isPreloaded = dom.radioSourcePreloaded.checked;
  const rawList = isPreloaded ? preloadedManifest : customUploadedFiles;
  
  const splitFilter = dom.validationSplitFilter.value;
  if (splitFilter === "all") {
    validationQueue = [...rawList];
  } else {
    validationQueue = rawList.filter(item => item.split === splitFilter);
  }
  
  if (validationQueue.length === 0) {
    dom.validationScanStatus.className = "scan-status-alert error";
    if (isPreloaded) {
      dom.validationScanStatus.innerText = `No files found matching split "${splitFilter}" in preloaded dataset.`;
    } else {
      dom.validationScanStatus.innerText = `No files found matching split "${splitFilter}" in uploaded folder. Select another folder.`;
    }
    dom.btnRunValidation.disabled = true;
    return;
  }
  
  dom.validationScanStatus.className = "scan-status-alert success";
  const sourceLabel = isPreloaded ? "preloaded dataset" : "custom folder";
  dom.validationScanStatus.innerText = `Loaded ${validationQueue.length.toLocaleString()} files from ${sourceLabel} (split: ${splitFilter}). Ready to validate.`;
  
  validationIndex = 0;
  validationProcessedCount = 0;
  
  const limitValue = parseInt(dom.validationLimit.value, 10);
  const totalToProcess = limitValue === -1 ? validationQueue.length : Math.min(limitValue, validationQueue.length);
  
  dom.valTotalCount.innerText = totalToProcess.toLocaleString();
  dom.valProcessedCount.innerText = "0";
  dom.validationProgressFill.style.width = "0%";
  
  dom.btnRunValidation.disabled = false;
  dom.btnRunValidation.innerText = "▶ Run Validation";
  dom.validationProgressPanel.style.display = "block";
}

async function startValidation() {
  if (validationQueue.length === 0) return;
  
  if (validationIndex === 0) {
    resetConfusionMatrix();
    validationProcessedCount = 0;
    validationSuccessCount = 0;
    validationFacesDetected = 0;
    validationStartTime = performance.now();
    validationTotalTime = 0;
    validationResultsData = [];
    dom.valLogsTbody.innerHTML = "";
    renderConfusionMatrix();
    renderValidationTable();
  }
  
  isValidationRunning = true;
  dom.validationRunStatus.innerText = "Status: Running...";
  dom.btnRunValidation.style.display = "none";
  dom.btnPauseValidation.style.display = "inline-flex";
  dom.btnCancelValidation.disabled = false;
  dom.btnSelectValidationDir.disabled = true;
  dom.validationLimit.disabled = true;
  dom.validationSplitFilter.disabled = true;
  dom.radioSourcePreloaded.disabled = true;
  dom.radioSourceCustom.disabled = true;
  
  processNextValidationImage();
}

function pauseValidation() {
  isValidationRunning = false;
  dom.validationRunStatus.innerText = "Status: Paused";
  dom.btnRunValidation.style.display = "inline-flex";
  dom.btnRunValidation.innerText = "▶ Resume";
  dom.btnPauseValidation.style.display = "none";
}

function resetValidationState() {
  isValidationRunning = false;
  validationIndex = 0;
  validationQueue = [];
  customUploadedFiles = [];
  validationProcessedCount = 0;
  validationSuccessCount = 0;
  validationFacesDetected = 0;
  validationResultsData = [];
  resetConfusionMatrix();
  
  dom.validationRunStatus.innerText = "Status: Idle";
  dom.validationSpeed.innerText = "-- ms / img";
  dom.validationProgressFill.style.width = "0%";
  dom.valProcessedCount.innerText = "0";
  dom.valTotalCount.innerText = "0";
  
  dom.btnRunValidation.style.display = "inline-flex";
  dom.btnRunValidation.innerText = "▶ Run Validation";
  dom.btnRunValidation.disabled = true;
  dom.btnPauseValidation.style.display = "none";
  dom.btnCancelValidation.disabled = true;
  
  dom.btnSelectValidationDir.disabled = false;
  dom.validationLimit.disabled = false;
  dom.validationSplitFilter.disabled = false;
  dom.radioSourcePreloaded.disabled = false;
  dom.radioSourceCustom.disabled = false;
  
  dom.validationProgressPanel.style.display = "none";
  dom.validationScanStatus.className = "scan-status-alert info";
  dom.validationScanStatus.innerText = "No directory selected. Ready to scan...";
  dom.validationDirInput.value = "";
  
  // Clear logs & tables
  dom.valStatAccuracy.innerText = "--%";
  dom.valStatProcessed.innerText = "0 / 0";
  dom.valStatDetectionRate.innerText = "--%";
  dom.valLogsTbody.innerHTML = `<tr><td colspan="6" class="text-center text-muted">Awaiting dataset run...</td></tr>`;
  
  // Reload/reprepare
  prepareDatasetQueue();
  renderConfusionMatrix();
  renderValidationTable();
}

async function processNextValidationImage() {
  if (!isValidationRunning) return;
  
  const limitValue = parseInt(dom.validationLimit.value, 10);
  const totalToProcess = limitValue === -1 ? validationQueue.length : Math.min(limitValue, validationQueue.length);
  
  if (validationIndex >= totalToProcess) {
    finalizeValidation();
    return;
  }
  
  const item = validationQueue[validationIndex];
  dom.valPreviewFilename.innerText = item.name;
  dom.valPreviewActual.innerText = item.trueLabel.toUpperCase();
  dom.valPreviewActual.style.background = EMOTION_METADATA[item.trueLabel].color;
  dom.valPreviewPredict.innerText = "Processing...";
  dom.valPreviewPredict.style.background = "rgba(255, 255, 255, 0.1)";
  
  const startTime = performance.now();
  
  try {
    // 1. Read & Decompress file into Image element
    let img;
    if (item.isPreloaded) {
      img = await loadImageFromUrl("/" + item.relativePath);
    } else {
      img = await loadImageFromFile(item.file);
    }
    
    // 2. Draw aspect ratio preserved preview thumbnail
    const pCtx = dom.validationPreviewCanvas.getContext("2d");
    pCtx.clearRect(0, 0, dom.validationPreviewCanvas.width, dom.validationPreviewCanvas.height);
    const wRatio = dom.validationPreviewCanvas.width / img.width;
    const hRatio = dom.validationPreviewCanvas.height / img.height;
    const ratio = Math.min(wRatio, hRatio);
    const w = img.width * ratio;
    const h = img.height * ratio;
    const x = (dom.validationPreviewCanvas.width - w) / 2;
    const y = (dom.validationPreviewCanvas.height - h) / 2;
    pCtx.drawImage(img, x, y, w, h);
    
    // 3. Paint on temporary offscreen canvas for MediaPipe (instant GPU upload)
    const offscreenCanvas = document.createElement("canvas");
    offscreenCanvas.width = img.width;
    offscreenCanvas.height = img.height;
    const offCtx = offscreenCanvas.getContext("2d");
    offCtx.drawImage(img, 0, 0);
    
    // 4. Run model inferences
    const results = await detectorInstance.detectImage(offscreenCanvas);
    
    let predictedLabel = EMOTIONS.NEUTRAL;
    let confidence = 0;
    
    let targetEngine = selectedModelEngine;
    if (targetEngine === "auto") {
      targetEngine = vitPipeline ? "vit" : (cnnSession ? "cnn" : (modelWeights ? "mlp" : "heuristics"));
    }
    
    let emotionRes = null;
    
    // 1. If target engine is ViT or CNN, evaluate directly bypassing MediaPipe face detection
    // because FER2013 preloaded images are already 48x48 cropped face squares, which is too small for MediaPipe's landmarker to find.
    if (targetEngine === "vit" && typeof classifyEmotionViT === "function" && vitPipeline) {
      try {
        emotionRes = await classifyEmotionViT(offscreenCanvas);
        if (emotionRes) {
          validationFacesDetected++; // Count as detected since model classified it
        }
      } catch (err) {
        console.error("Validation ViT inference error:", err);
        addLog(`ViT run fail on ${item.name}: ${err.message}`, "error");
      }
    } else if (targetEngine === "cnn" && typeof classifyEmotionCNN === "function" && cnnSession) {
      try {
        emotionRes = await classifyEmotionCNN(offscreenCanvas);
        if (emotionRes) {
          validationFacesDetected++; // Count as detected since model classified it
        }
      } catch (err) {
        console.error("Validation CNN inference error:", err);
        addLog(`CNN run fail on ${item.name}: ${err.message}`, "error");
      }
    }
    
    // 2. If running MLP/Baseline, resolve using MediaPipe blendshapes
    if (!emotionRes) {
      if (results.faceResult && results.faceResult.faceBlendshapes && results.faceResult.faceBlendshapes.length > 0) {
        validationFacesDetected++;
        emotionRes = classifyEmotion(results.faceResult, 0);
      }
    }
    
    if (emotionRes) {
      predictedLabel = emotionRes.dominantEmotion;
      confidence = emotionRes.confidence;
    }
    
    const elapsed = performance.now() - startTime;
    validationTotalTime += elapsed;
    
    const match = predictedLabel === item.trueLabel;
    if (match) {
      validationSuccessCount++;
    }
    validationProcessedCount++;
    
    // Update confusion matrix data
    confusionMatrix[item.trueLabel][predictedLabel]++;
    
    // Render current prediction badge
    dom.valPreviewPredict.innerText = predictedLabel.toUpperCase();
    dom.valPreviewPredict.style.background = EMOTION_METADATA[predictedLabel].color;
    
    // Log evaluation detail
    appendValidationLog(item.name, item.relativePath, item.trueLabel, predictedLabel, confidence, match);
    
    // Update dashboard summary metrics
    const overallAccuracy = Math.round((validationSuccessCount / validationProcessedCount) * 100);
    dom.valStatAccuracy.innerText = `${overallAccuracy}%`;
    dom.valStatProcessed.innerText = `${validationProcessedCount} / ${totalToProcess}`;
    
    const faceDetectionRate = Math.round((validationFacesDetected / validationProcessedCount) * 100);
    dom.valStatDetectionRate.innerText = `${faceDetectionRate}%`;
    
    // Redraw matrix and per-emotion counts
    renderConfusionMatrix();
    renderValidationTable();
    
    // Move progress bar
    const progressPercent = (validationProcessedCount / totalToProcess) * 100;
    dom.validationProgressFill.style.width = `${progressPercent}%`;
    dom.valProcessedCount.innerText = validationProcessedCount;
    
    // Render average speed
    const avgTime = Math.round(validationTotalTime / validationProcessedCount);
    dom.validationSpeed.innerText = `${avgTime} ms / img`;
    
  } catch (err) {
    console.error("Error processing validation file:", err);
  }
  
  validationIndex++;
  
  if (isValidationRunning) {
    // Delay slightly to give browser event loop breathing room for CSS paints/UI events
    setTimeout(processNextValidationImage, 15);
  }
}

function finalizeValidation() {
  isValidationRunning = false;
  dom.validationRunStatus.innerText = "Status: Completed";
  dom.btnRunValidation.style.display = "inline-flex";
  dom.btnRunValidation.innerText = "▶ Restart";
  dom.btnRunValidation.disabled = false;
  dom.btnPauseValidation.style.display = "none";
  dom.btnCancelValidation.disabled = false;
  
  dom.btnSelectValidationDir.disabled = false;
  dom.validationLimit.disabled = false;
  
  addLog(`Validation complete. Evaluated ${validationProcessedCount} images. Accuracy: ${dom.valStatAccuracy.innerText}`, "system");
}

function renderConfusionMatrix() {
  const grid = dom.confusionMatrixGrid;
  grid.innerHTML = "";
  
  // Header: Act\Pred
  const corner = document.createElement("div");
  corner.className = "matrix-cell header-cell";
  corner.innerText = "Act\\Pred";
  grid.appendChild(corner);
  
  for (const pred of VALIDATION_EMOTIONS) {
    const header = document.createElement("div");
    header.className = "matrix-cell header-cell";
    header.innerText = pred.charAt(0).toUpperCase() + pred.slice(1, 4);
    grid.appendChild(header);
  }
  
  // Matrix Rows
  for (const act of VALIDATION_EMOTIONS) {
    const rowLabel = document.createElement("div");
    rowLabel.className = "matrix-cell header-cell";
    rowLabel.innerText = act.charAt(0).toUpperCase() + act.slice(1, 4);
    grid.appendChild(rowLabel);
    
    let rowTotal = 0;
    for (const pred of VALIDATION_EMOTIONS) {
      rowTotal += confusionMatrix[act][pred];
    }
    
    for (const pred of VALIDATION_EMOTIONS) {
      const count = confusionMatrix[act][pred];
      const cell = document.createElement("div");
      cell.className = "matrix-cell data-cell";
      cell.innerText = count || "-";
      
      if (count > 0) {
        const rate = rowTotal > 0 ? (count / rowTotal) : 0;
        cell.style.background = `rgba(139, 92, 246, ${Math.min(0.85, 0.15 + rate * 0.7)})`;
        cell.style.color = "#FFFFFF";
        cell.title = `Actual: ${act}, Predicted: ${pred} (${count} samples, ${Math.round(rate * 100)}%)`;
      }
      grid.appendChild(cell);
    }
  }
}

function renderValidationTable() {
  const tbody = dom.valBreakdownTbody;
  tbody.innerHTML = "";
  
  const stats = {};
  for (const emotion of VALIDATION_EMOTIONS) {
    stats[emotion] = { total: 0, tp: 0, fp: 0 };
  }
  
  for (const act of VALIDATION_EMOTIONS) {
    for (const pred of VALIDATION_EMOTIONS) {
      const count = confusionMatrix[act][pred];
      stats[act].total += count;
      if (act === pred) {
        stats[act].tp += count;
      } else {
        stats[pred].fp += count;
      }
    }
  }
  
  for (const emotion of VALIDATION_EMOTIONS) {
    const s = stats[emotion];
    const meta = EMOTION_METADATA[emotion];
    const rateText = s.total > 0 ? `${Math.round((s.tp / s.total) * 100)}%` : "--";
    
    const row = document.createElement("tr");
    row.innerHTML = `
      <td><span style="color: ${meta.color}; font-weight: 600;">${meta.emoji} ${meta.label}</span></td>
      <td>${s.total}</td>
      <td>${s.tp}</td>
      <td>${s.fp}</td>
      <td style="font-weight: bold; color: ${s.total > 0 ? '#FFFFFF' : 'var(--text-muted)'}">${rateText}</td>
    `;
    tbody.appendChild(row);
  }
}

function appendValidationLog(filename, relativePath, trueLabel, predictedLabel, confidence, match) {
  const tbody = dom.valLogsTbody;
  
  if (validationProcessedCount === 1) {
    tbody.innerHTML = "";
    validationLogsHistory = []; // Reset logs history for a new run
  }
  
  const trueMeta = EMOTION_METADATA[trueLabel];
  const predMeta = EMOTION_METADATA[predictedLabel];
  const confPercent = `${Math.round(confidence * 100)}%`;

  // Append to logs history for copy functionality
  validationLogsHistory.push({
    filename,
    relativePath,
    trueLabel,
    predictedLabel,
    confidence: confPercent,
    match: match ? "Match" : "Mismatch"
  });
  
  const row = document.createElement("tr");
  const statusBadge = match 
    ? `<span class="status-badge correct">✅ Match</span>` 
    : `<span class="status-badge incorrect">❌ Mismatch</span>`;
    
  row.innerHTML = `
    <td><canvas class="val-thumb-canvas" width="28" height="28"></canvas></td>
    <td class="text-truncate" style="max-width: 140px;" title="${relativePath}">${relativePath}</td>
    <td><span style="color: ${trueMeta.color}">${trueMeta.emoji} ${trueMeta.label}</span></td>
    <td class="${match ? 'match' : 'mismatch'}">${predMeta.emoji} ${predMeta.label}</td>
    <td>${confPercent}</td>
    <td>${statusBadge}</td>
  `;
  
  tbody.insertBefore(row, tbody.firstChild);
  
  if (tbody.children.length > 50) {
    tbody.removeChild(tbody.lastChild);
  }
  
  // Copy canvas thumbnail
  const thumbCanvas = row.querySelector(".val-thumb-canvas");
  const thumbCtx = thumbCanvas.getContext("2d");
  thumbCtx.drawImage(dom.validationPreviewCanvas, 0, 0, 28, 28);
}

// --- Event Listeners Setup ---
function setupEventListeners() {
  // Tab buttons
  dom.tabBtnCamera.addEventListener("click", () => switchTab("camera"));
  dom.tabBtnUpload.addEventListener("click", () => switchTab("upload"));
  dom.tabBtnValidation.addEventListener("click", () => switchTab("validation"));

  // Webcam buttons
  dom.btnStartCamera.addEventListener("click", startCameraStream);
  dom.btnStopCamera.addEventListener("click", stopCameraStream);

  // File Upload buttons & inputs
  dom.btnBrowseFile.addEventListener("click", (e) => {
    e.stopPropagation(); // Prevent bubbling up to dropZone (which would trigger the picker twice)
    triggerFileBrowser();
  });
  dom.fileInput.addEventListener("click", (e) => {
    e.stopPropagation(); // Prevent bubbling up when click() is programmatically called
  });
  dom.fileInput.addEventListener("change", handleFileSelect);
  dom.btnResetUpload.addEventListener("click", () => {
    resetUploadUI();
    resetAnalysisUI();
    addLog("Upload image cleared.", "system");
  });

  // Drag & drop zones
  dom.dropZone.addEventListener("dragover", handleDragOver);
  dom.dropZone.addEventListener("dragleave", handleDragLeave);
  dom.dropZone.addEventListener("drop", handleDrop);
  dom.dropZone.addEventListener("click", triggerFileBrowser);

  // Validation Suite controls
  dom.radioSourcePreloaded.addEventListener("change", () => toggleDatasetSource("preloaded"));
  dom.radioSourceCustom.addEventListener("change", () => toggleDatasetSource("custom"));
  dom.validationLimit.addEventListener("change", prepareDatasetQueue);
  dom.validationSplitFilter.addEventListener("change", prepareDatasetQueue);

  dom.btnSelectValidationDir.addEventListener("click", () => dom.validationDirInput.click());
  dom.validationDirInput.addEventListener("change", handleValidationDirSelect);
  dom.btnRunValidation.addEventListener("click", startValidation);
  dom.btnPauseValidation.addEventListener("click", pauseValidation);
  dom.btnCancelValidation.addEventListener("click", resetValidationState);

  // Drawing settings checkboxes
  dom.chkShowFace.addEventListener("change", handleOverlayCheckboxChange);
  dom.chkShowPose.addEventListener("change", handleOverlayCheckboxChange);

  // Model engine selectors (synchronized across tabs)
  function handleModelEngineChange(engineValue) {
    dom.selectModelEngine.value = engineValue;
    dom.validationModelEngine.value = engineValue;
    
    setSelectedModelEngine(engineValue);
    lastEmotionResult = null; // Clear cached ViT result to force fresh evaluation
    
    // If we have cached inference results, re-evaluate and draw immediately
    if (cachedLastResults.faceResult || cachedLastResults.poseResult) {
      const source = activeTab === "camera" ? dom.webcamVideo : dom.uploadPreview;
      updateAnalysisResults(cachedLastResults, source);
      drawVisualOverlays(activeTab === "camera" ? dom.webcamCanvas : dom.uploadCanvas, cachedLastResults);
    } else {
      let active = engineValue;
      if (active === "auto") {
        active = vitPipeline ? "vit" : (cnnSession ? "cnn" : (modelWeights ? "mlp" : "heuristics"));
      }
      if (active === "vit" && !vitPipeline) active = "heuristics";
      if (active === "cnn" && !cnnSession) active = "heuristics";
      if (active === "mlp" && !modelWeights) active = "heuristics";
      updateActiveModelLabel(active);
    }
  }

  dom.selectModelEngine.addEventListener("change", (e) => handleModelEngineChange(e.target.value));
  dom.validationModelEngine.addEventListener("change", (e) => handleModelEngineChange(e.target.value));

  // Copy System Logs to Clipboard
  dom.btnCopySystemLogs.addEventListener("click", () => {
    const container = document.getElementById("logs-container");
    if (!container) return;
    const entries = Array.from(container.children)
      .map(el => el.innerText)
      .join("\n");
    
    navigator.clipboard.writeText(entries).then(() => {
      const originalText = dom.btnCopySystemLogs.innerText;
      dom.btnCopySystemLogs.innerText = "✔ Copied!";
      setTimeout(() => {
        dom.btnCopySystemLogs.innerText = originalText;
      }, 1500);
    }).catch(err => {
      console.error("Failed to copy system logs:", err);
    });
  });

  // Copy Validation Logs as CSV to Clipboard
  dom.btnCopyValLogs.addEventListener("click", () => {
    if (validationLogsHistory.length === 0) {
      alert("No validation logs available to copy. Please run the validation suite first.");
      return;
    }
    
    const csvContent = [
      "File Name,Relative Path,Actual Emotion,Predicted Emotion,Confidence,Result",
      ...validationLogsHistory.map(log => 
        `"${log.filename}","${log.relativePath}","${log.trueLabel}","${log.predictedLabel}","${log.confidence}","${log.match}"`
      )
    ].join("\n");
    
    navigator.clipboard.writeText(csvContent).then(() => {
      const originalText = dom.btnCopyValLogs.innerText;
      dom.btnCopyValLogs.innerText = "✔ Copied!";
      setTimeout(() => {
        dom.btnCopyValLogs.innerText = originalText;
      }, 1500);
    }).catch(err => {
      console.error("Failed to copy validation CSV:", err);
    });
  });
}

// Start
setupEventListeners();
initializeApp();
