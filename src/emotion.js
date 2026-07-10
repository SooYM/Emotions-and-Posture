import { pipeline, env, RawImage } from '@huggingface/transformers';

export const EMOTIONS = {
  HAPPY: "happy",
  SAD: "sad",
  ANXIETY: "anxiety",
  ANGRY: "angry",
  SURPRISED: "surprised",
  DISGUST: "disgust",
  NEUTRAL: "neutral"
};

export const EMOTION_METADATA = {
  [EMOTIONS.HAPPY]: { label: "Happy", emoji: "😊", color: "#10B981" },     // Emerald green
  [EMOTIONS.SAD]: { label: "Sad", emoji: "😢", color: "#3B82F6" },         // Blue
  [EMOTIONS.ANXIETY]: { label: "Anxiety", emoji: "😰", color: "#8B5CF6" }, // Purple
  [EMOTIONS.ANGRY]: { label: "Angry", emoji: "😡", color: "#EF4444" },     // Red
  [EMOTIONS.SURPRISED]: { label: "Surprised", emoji: "😲", color: "#F59E0B" }, // Amber/Orange
  [EMOTIONS.DISGUST]: { label: "Disgust", emoji: "🤢", color: "#84CC16" },   // Lime
  [EMOTIONS.NEUTRAL]: { label: "Neutral", emoji: "😐", color: "#6B7280" }    // Gray
};

export let modelWeights = null;
export let vitPipeline = null;
export let cnnSession = null;
export let modelLoadErrors = {
  vitWebGPU: null,
  vitCPU: null,
  cnnWebGPU: null,
  cnnCPU: null,
  mlp: null
};

// Map the Hugging Face classes (both name strings and fallback index names) to our internal EMOTIONS labels
const HF_TO_INTERNAL = {
  "anger": EMOTIONS.ANGRY,
  "angry": EMOTIONS.ANGRY,
  "label_0": EMOTIONS.ANGRY,
  "disgust": EMOTIONS.DISGUST,
  "label_1": EMOTIONS.DISGUST,
  "fear": EMOTIONS.ANXIETY,
  "anxiety": EMOTIONS.ANXIETY,
  "label_2": EMOTIONS.ANXIETY,
  "happy": EMOTIONS.HAPPY,
  "label_3": EMOTIONS.HAPPY,
  "neutral": EMOTIONS.NEUTRAL,
  "label_4": EMOTIONS.NEUTRAL,
  "sad": EMOTIONS.SAD,
  "label_5": EMOTIONS.SAD,
  "surprise": EMOTIONS.SURPRISED,
  "surprised": EMOTIONS.SURPRISED,
  "label_6": EMOTIONS.SURPRISED
};

// Set local model path and WASM paths for offline Hugging Face pipelines
if (typeof window !== 'undefined' && window.location) {
  env.localModelPath = window.location.origin + '/';
} else {
  env.localModelPath = '/';
}
if (typeof window !== 'undefined' && window.location) {
  env.backends.onnx.wasm.wasmPaths = window.location.origin + '/';
  if (window.ort) {
    window.ort.env.wasm.wasmPaths = window.location.origin + '/';
  }
} else {
  env.backends.onnx.wasm.wasmPaths = '/';
}
env.allowLocalModels = true;
env.allowRemoteModels = false; // Run strictly offline/locally
env.useBrowserCache = false;   // Disable custom browser cache storage to prevent loaded 404 cache poisoning

async function checkWebGPUSupport() {
  const checkPromise = (async () => {
    if (typeof navigator === 'undefined' || !navigator.gpu) return false;
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return false;
      const device = await adapter.requestDevice();
      if (!device) return false;
      device.destroy();
      return true;
    } catch (e) {
      return false;
    }
  })();
  
  return Promise.race([
    checkPromise,
    new Promise(resolve => setTimeout(() => resolve(false), 2000))
  ]);
}

export async function loadModelWeights(onModelLoaded = null) {
  const mlpPromise = (async () => {
    try {
      const response = await fetch("/model_weights.json");
      if (response.ok) {
        modelWeights = await response.json();
        console.log("Successfully loaded machine-learned emotion classification weights.", modelWeights);
        if (onModelLoaded) onModelLoaded("mlp");
      } else {
        modelLoadErrors.mlp = `Weights file status: ${response.status} ${response.statusText}`;
        console.warn("Pre-trained model weights JSON not found. Defaulting to rule-based heuristics.");
      }
    } catch (err) {
      modelLoadErrors.mlp = err.message;
      console.warn("Failed to retrieve model weights. Defaulting to rule-based heuristics.", err);
    }
  })();

  const vitPromise = (async () => {
    try {
      console.log("Checking WebGPU support for Hugging Face ViT model...");
      const hasWebGPU = await checkWebGPUSupport();
      if (hasWebGPU) {
        console.log("Initializing Hugging Face Vision Transformer (ViT) on WebGPU...");
        vitPipeline = await pipeline('image-classification', 'vit_onnx', {
          device: 'webgpu', // Use WebGPU for extremely fast client-side inference
          quantized: false  // Do not search for quantized versions, load model.onnx directly
        });
        console.log("Pre-trained Hugging Face ViT model loaded successfully with WebGPU.");
        if (onModelLoaded) onModelLoaded("vit");
      } else {
        throw new Error("WebGPU is not supported or device creation timed out.");
      }
    } catch (err) {
      modelLoadErrors.vitWebGPU = err.message;
      console.warn("Failed to load ViT on WebGPU. Retrying on CPU...", err);
      try {
        vitPipeline = await pipeline('image-classification', 'vit_onnx', {
          device: 'wasm', // Fallback to WebAssembly CPU
          quantized: false
        });
        console.log("Pre-trained Hugging Face ViT model loaded successfully on CPU.");
        if (onModelLoaded) onModelLoaded("vit");
      } catch (cpuErr) {
        modelLoadErrors.vitCPU = cpuErr.message;
        console.error("Failed to load ViT model. Emotion classification will fall back to MLP/Heuristics.", cpuErr);
      }
    }
  })();

  const cnnPromise = (async () => {
    try {
      console.log("Initializing Custom PyTorch CNN (FERNet) via ONNX Runtime WASM...");
      cnnSession = await window.ort.InferenceSession.create('/cnn_onnx/model.onnx', {
        executionProviders: ['wasm']
      });
      console.log("Custom PyTorch CNN (FERNet) ONNX model loaded successfully on WASM.");
      if (onModelLoaded) onModelLoaded("cnn");
    } catch (err) {
      modelLoadErrors.cnnCPU = err.message;
      console.error("Failed to load CNN model on WASM:", err);
    }
  })();

  // Execute all loading processes in parallel
  await Promise.all([mlpPromise, vitPromise, cnnPromise]);
}

export async function classifyEmotionViT(canvasOrImage) {
  if (!vitPipeline) {
    return null;
  }
  let input = canvasOrImage;
  if (canvasOrImage instanceof HTMLCanvasElement) {
    input = RawImage.fromCanvas(canvasOrImage);
  }
  const output = await vitPipeline(input);
  const scores = {};
  for (const e of Object.values(EMOTIONS)) {
    scores[e] = 0;
  }
  
  let dominantEmotion = EMOTIONS.NEUTRAL;
  let maxScore = -1;
  
  for (const item of output) {
    const internalKey = HF_TO_INTERNAL[item.label.toLowerCase()];
    if (internalKey) {
      scores[internalKey] = item.score;
      if (item.score > maxScore) {
        maxScore = item.score;
        dominantEmotion = internalKey;
      }
    }
  }
  
  return {
    dominantEmotion,
    confidence: maxScore,
    scores,
    activeModel: "vit"
  };
}

function preprocessCanvasTo48x48Grayscale(canvasOrImage) {
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = 48;
  tempCanvas.height = 48;
  const tempCtx = tempCanvas.getContext("2d");
  
  tempCtx.drawImage(canvasOrImage, 0, 0, 48, 48);
  
  const imgData = tempCtx.getImageData(0, 0, 48, 48);
  const data = imgData.data;
  
  const float32Array = new Float32Array(48 * 48);
  
  for (let i = 0; i < 48 * 48; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    
    // Grayscale conversion: Y = 0.299R + 0.587G + 0.114B
    const gray = 0.299 * r + 0.587 * g + 0.114 * b;
    
    // Normalization matching transforms.Normalize((0.5,), (0.5,)) -> (gray - 127.5) / 127.5
    float32Array[i] = (gray - 127.5) / 127.5;
  }
  
  return float32Array;
}

export async function classifyEmotionCNN(canvasOrImage) {
  if (!cnnSession) {
    return null;
  }
  try {
    const float32Data = preprocessCanvasTo48x48Grayscale(canvasOrImage);
    const inputTensor = new window.ort.Tensor('float32', float32Data, [1, 1, 48, 48]);
    
    const outputMap = await cnnSession.run({ 'input': inputTensor });
    const outputTensor = outputMap['output'];
    const logits = outputTensor.data;
    
    // Apply Softmax activation
    const maxLogit = Math.max(...logits);
    const exps = Array.from(logits).map(x => Math.exp(x - maxLogit));
    const sumExps = exps.reduce((sum, val) => sum + val, 0);
    const probabilities = exps.map(x => sumExps > 0 ? (x / sumExps) : 0);
    
    // Match alphabetical PyTorch Class Indices:
    // {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
    const EMOTION_ORDER = [
      EMOTIONS.ANGRY,
      EMOTIONS.DISGUST,
      EMOTIONS.ANXIETY, // fear -> anxiety
      EMOTIONS.HAPPY,
      EMOTIONS.NEUTRAL,
      EMOTIONS.SAD,
      EMOTIONS.SURPRISED // surprise -> surprised
    ];
    
    const scores = {};
    let dominantEmotion = EMOTIONS.NEUTRAL;
    let maxScore = -1;
    
    for (let i = 0; i < EMOTION_ORDER.length; i++) {
      const emotion = EMOTION_ORDER[i];
      scores[emotion] = probabilities[i];
      if (probabilities[i] > maxScore) {
        maxScore = probabilities[i];
        dominantEmotion = emotion;
      }
    }
    
    return {
      dominantEmotion,
      confidence: maxScore,
      scores,
      activeModel: "cnn"
    };
  } catch (err) {
    console.error("CNN inference failure:", err);
    return null;
  }
}

export let selectedModelEngine = "auto";
export function setSelectedModelEngine(engine) {
  selectedModelEngine = engine;
  console.log(`Emotion classifier model engine updated to: ${engine}`);
}

export function classifyEmotion(faceResult, faceIndex = 0) {
  // Default values when no face is detected
  if (!faceResult || !faceResult.faceBlendshapes || faceResult.faceBlendshapes.length <= faceIndex) {
    const defaultScores = {};
    for (const e of Object.values(EMOTIONS)) {
      defaultScores[e] = 0;
    }
    return {
      dominantEmotion: EMOTIONS.NEUTRAL,
      confidence: 0,
      scores: defaultScores,
      activeModel: "heuristics"
    };
  }

  let activeModel = "heuristics";
  let targetEngine = selectedModelEngine;

  // Auto-Select resolves to MLP first on synchronous calls if weights are loaded
  if (targetEngine === "auto") {
    if (modelWeights && modelWeights.type === "mlp") {
      targetEngine = "mlp";
    } else {
      targetEngine = "heuristics";
    }
  }

  let res = null;

  if (targetEngine === "mlp" && modelWeights && modelWeights.type === "mlp") {
    res = runMLPInference(faceResult, faceIndex);
    if (res) activeModel = "mlp";
  }

  if (!res) {
    res = runHeuristicInference(faceResult, faceIndex);
    activeModel = "heuristics";
  }

  return {
    ...res,
    activeModel
  };
}

function runMLPInference(faceResult, faceIndex) {
  const categories = faceResult.faceBlendshapes[faceIndex].categories;
  const shapes = {};
  for (const item of categories) {
    shapes[item.categoryName] = item.score;
  }

  const input = modelWeights.features.map(name => shapes[name] || 0);
  const logitScores = [];

  // Support both 1-hidden-layer (old) and 2-hidden-layer (new) MLPs dynamically
  const isTwoLayer = modelWeights.w3 && modelWeights.b3;

  if (isTwoLayer) {
    // 2-layer MLP
    // Layer 1: Hidden layer 1 (ReLU)
    const hiddenSize1 = modelWeights.b1.length;
    const hidden1 = new Array(hiddenSize1);
    for (let h = 0; h < hiddenSize1; h++) {
      let score = modelWeights.b1[h];
      for (let f = 0; f < input.length; f++) {
        score += input[f] * modelWeights.w1[f][h];
      }
      hidden1[h] = Math.max(0, score);
    }
    
    // Layer 2: Hidden layer 2 (ReLU)
    const hiddenSize2 = modelWeights.b2.length;
    const hidden2 = new Array(hiddenSize2);
    for (let h = 0; h < hiddenSize2; h++) {
      let score = modelWeights.b2[h];
      for (let h1 = 0; h1 < hiddenSize1; h1++) {
        score += hidden1[h1] * modelWeights.w2[h1][h];
      }
      hidden2[h] = Math.max(0, score);
    }
    
    // Layer 3: Output logits
    for (let c = 0; c < modelWeights.classes.length; c++) {
      const className = modelWeights.classes[c];
      let score = modelWeights.b3[c];
      for (let h2 = 0; h2 < hiddenSize2; h2++) {
        score += hidden2[h2] * modelWeights.w3[h2][c];
      }
      logitScores.push({ className, score });
    }
  } else {
    // 1-layer MLP
    // Layer 1: Hidden layer (ReLU)
    const hiddenSize = modelWeights.b1.length;
    const hidden = new Array(hiddenSize);
    for (let h = 0; h < hiddenSize; h++) {
      let score = modelWeights.b1[h];
      for (let f = 0; f < input.length; f++) {
        score += input[f] * modelWeights.w1[f][h];
      }
      hidden[h] = Math.max(0, score);
    }
    
    // Layer 2: Output logits
    for (let c = 0; c < modelWeights.classes.length; c++) {
      const className = modelWeights.classes[c];
      let score = modelWeights.b2[c];
      for (let h = 0; h < hiddenSize; h++) {
        score += hidden[h] * modelWeights.w2[h][c];
      }
      logitScores.push({ className, score });
    }
  }
  
  // Softmax normalization
  const expScores = logitScores.map(item => Math.exp(item.score));
  const sumExp = expScores.reduce((sum, val) => sum + val, 0);
  const scores = {};
  for (let i = 0; i < logitScores.length; i++) {
    scores[logitScores[i].className] = sumExp > 0 ? (expScores[i] / sumExp) : 0;
  }
  
  let dominantEmotion = EMOTIONS.NEUTRAL;
  let maxScore = -1;
  for (const emotion in scores) {
    if (scores[emotion] > maxScore) {
      maxScore = scores[emotion];
      dominantEmotion = emotion;
    }
  }
  
  return {
    dominantEmotion,
    confidence: scores[dominantEmotion],
    scores
  };
}

function runHeuristicInference(faceResult, faceIndex) {
  // Rule-based heuristics cleared as requested.
  // Returns a default Neutral state when no machine-learning model is selected or active.
  const defaultScores = {};
  for (const e of Object.values(EMOTIONS)) {
    defaultScores[e] = e === EMOTIONS.NEUTRAL ? 1.0 : 0.0;
  }
  return {
    dominantEmotion: EMOTIONS.NEUTRAL,
    confidence: 1.0,
    scores: defaultScores
  };
}
