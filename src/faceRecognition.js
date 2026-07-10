/**
 * Face Recognition Module
 * Uses @vladmandic/face-api for face descriptor extraction and matching.
 * Registered faces are stored in localStorage for persistence across sessions.
 */
import * as faceapi from '@vladmandic/face-api';

const MODEL_URL = '/face-api-models';
const MATCH_THRESHOLD = 0.55; // Euclidean distance threshold (lower = stricter)
const STORAGE_KEY = 'faceRecognition_registeredFaces';

let isModelLoaded = false;
let registeredFaces = []; // Array of { name, descriptors: [Float32Array...] }
let faceMatcher = null;

/**
 * Initialize face-api models required for recognition.
 * Loads SsdMobilenetv1 (detection), FaceLandmark68TinyNet (alignment), FaceRecognitionNet (128-dim embeddings).
 */
export async function initFaceRecognition(onStatus = null) {
  if (isModelLoaded) return true;
  try {
    if (onStatus) onStatus('Loading face recognition models...');
    await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
    await faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL);
    await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
    isModelLoaded = true;
    
    // Load any previously registered faces from localStorage
    loadRegisteredFaces();
    rebuildFaceMatcher();
    
    if (onStatus) onStatus('Face recognition models loaded.');
    return true;
  } catch (err) {
    console.error('Failed to load face recognition models:', err);
    if (onStatus) onStatus('Face recognition model load failed.');
    return false;
  }
}

/**
 * Detect all faces in an image/canvas and return their 128-dim descriptors.
 * @param {HTMLCanvasElement|HTMLVideoElement|HTMLImageElement} input
 * @returns {Array<{detection: object, descriptor: Float32Array}>}
 */
export async function detectFaceDescriptors(input) {
  if (!isModelLoaded) return [];
  const detections = await faceapi
    .detectAllFaces(input, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 }))
    .withFaceLandmarks(true) // useTinyModel = true
    .withFaceDescriptors();
  return detections;
}

/**
 * Register a new face from a video/canvas/image source.
 * Captures multiple descriptors from the primary detected face.
 * @param {string} name - Person's name
 * @param {HTMLCanvasElement|HTMLVideoElement} source
 * @returns {{ success: boolean, message: string }}
 */
export async function registerFace(name, source) {
  if (!isModelLoaded) {
    return { success: false, message: 'Face recognition models not loaded.' };
  }
  if (!name || !name.trim()) {
    return { success: false, message: 'Please enter a valid name.' };
  }
  
  const trimmedName = name.trim();
  
  const detections = await detectFaceDescriptors(source);
  if (!detections || detections.length === 0) {
    return { success: false, message: 'No face detected. Please face the camera directly.' };
  }
  
  // Use the largest (closest) detected face
  const primaryFace = detections.reduce((best, det) => {
    const box = det.detection.box;
    const area = box.width * box.height;
    const bestArea = best.detection.box.width * best.detection.box.height;
    return area > bestArea ? det : best;
  });
  
  const descriptor = primaryFace.descriptor;
  
  // Check if this person already exists — add descriptor to existing entry
  const existingIdx = registeredFaces.findIndex(
    f => f.name.toLowerCase() === trimmedName.toLowerCase()
  );
  
  if (existingIdx >= 0) {
    registeredFaces[existingIdx].descriptors.push(Array.from(descriptor));
    // Keep max 10 descriptors per person for diversity
    if (registeredFaces[existingIdx].descriptors.length > 10) {
      registeredFaces[existingIdx].descriptors.shift();
    }
  } else {
    registeredFaces.push({
      name: trimmedName,
      descriptors: [Array.from(descriptor)]
    });
  }
  
  saveRegisteredFaces();
  rebuildFaceMatcher();
  
  const count = existingIdx >= 0 
    ? registeredFaces[existingIdx].descriptors.length 
    : 1;
  
  return { 
    success: true, 
    message: `Face registered for "${trimmedName}" (${count} sample${count > 1 ? 's' : ''}).`
  };
}

/**
 * Match detected faces against registered faces.
 * @param {Array} detections - Output from detectFaceDescriptors()
 * @returns {Array<{name: string, distance: number, box: object}>}
 */
export function matchFaces(detections) {
  if (!faceMatcher || !detections || detections.length === 0) return [];
  
  return detections.map(det => {
    const match = faceMatcher.findBestMatch(det.descriptor);
    return {
      name: match.label === 'unknown' ? null : match.label,
      distance: match.distance,
      box: det.detection.box
    };
  });
}

/**
 * Remove a registered person by name.
 * @param {string} name
 */
export function unregisterFace(name) {
  registeredFaces = registeredFaces.filter(
    f => f.name.toLowerCase() !== name.toLowerCase()
  );
  saveRegisteredFaces();
  rebuildFaceMatcher();
}

/**
 * Clear all registered faces.
 */
export function clearAllFaces() {
  registeredFaces = [];
  faceMatcher = null;
  saveRegisteredFaces();
}

/**
 * Get a list of all registered face names and their sample counts.
 * @returns {Array<{name: string, sampleCount: number}>}
 */
export function getRegisteredFaces() {
  return registeredFaces.map(f => ({
    name: f.name,
    sampleCount: f.descriptors.length
  }));
}

/**
 * Check if face recognition models are loaded and ready.
 */
export function isFaceRecognitionReady() {
  return isModelLoaded;
}

/**
 * Check if any faces are registered.
 */
export function hasRegisteredFaces() {
  return registeredFaces.length > 0;
}

// --- Internal Helpers ---

function rebuildFaceMatcher() {
  if (registeredFaces.length === 0) {
    faceMatcher = null;
    return;
  }
  
  const labeledDescriptors = registeredFaces.map(face => {
    const descriptors = face.descriptors.map(d => new Float32Array(d));
    return new faceapi.LabeledFaceDescriptors(face.name, descriptors);
  });
  
  faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, MATCH_THRESHOLD);
}

function saveRegisteredFaces() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(registeredFaces));
  } catch (err) {
    console.error('Failed to save registered faces to localStorage:', err);
  }
}

function loadRegisteredFaces() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      registeredFaces = JSON.parse(stored);
      console.log(`Loaded ${registeredFaces.length} registered face(s) from localStorage.`);
    }
  } catch (err) {
    console.error('Failed to load registered faces from localStorage:', err);
    registeredFaces = [];
  }
}
