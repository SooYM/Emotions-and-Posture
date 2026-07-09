import { FilesetResolver, FaceLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision";

// Model URLs from Google Storage CDN
const FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task";

class MediaPipeDetector {
  constructor() {
    this.faceLandmarker = null;
    this.poseLandmarker = null;
    this.isLoaded = false;
    this.currentMode = null; // 'IMAGE' or 'VIDEO'
  }

  async init(onProgress) {
    try {
      if (onProgress) onProgress("Initializing Fileset Resolver...");
      
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );

      if (onProgress) onProgress("Loading Face Landmarker model (~5MB)...");
      this.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: FACE_MODEL_URL,
          delegate: "GPU"
        },
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: false,
        runningMode: "IMAGE",
        numFaces: 4
      });

      if (onProgress) onProgress("Loading Pose Landmarker model (~6MB)...");
      this.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: POSE_MODEL_URL,
          delegate: "GPU"
        },
        runningMode: "IMAGE",
        numPoses: 1
      });

      this.currentMode = "IMAGE";
      this.isLoaded = true;
      if (onProgress) onProgress("All models loaded successfully!");
      return true;
    } catch (error) {
      console.error("Failed to initialize MediaPipe models:", error);
      throw error;
    }
  }

  async setRunningMode(mode) {
    if (!this.isLoaded) return;
    if (this.currentMode === mode) return;

    // mode is either 'IMAGE' or 'VIDEO'
    await this.faceLandmarker.setOptions({ runningMode: mode });
    await this.poseLandmarker.setOptions({ runningMode: mode });
    this.currentMode = mode;
  }

  async detectImage(imageElement) {
    if (!this.isLoaded) throw new Error("Detector not initialized");
    await this.setRunningMode("IMAGE");

    const faceResult = this.faceLandmarker.detect(imageElement);
    const poseResult = this.poseLandmarker.detect(imageElement);

    return { faceResult, poseResult };
  }

  detectVideoFrame(videoElement, timestamp) {
    if (!this.isLoaded) throw new Error("Detector not initialized");
    
    // We expect the running mode to be updated to VIDEO beforehand for performance,
    // but we check it here just in case.
    if (this.currentMode !== "VIDEO") {
      this.setRunningMode("VIDEO");
    }

    const faceResult = this.faceLandmarker.detectForVideo(videoElement, timestamp);
    const poseResult = this.poseLandmarker.detectForVideo(videoElement, timestamp);

    return { faceResult, poseResult };
  }
}

export const detectorInstance = new MediaPipeDetector();
