import { classifyEmotion, EMOTION_METADATA } from "./emotion.js";

// Visualizer helper to draw face and pose landmarks on a 2D canvas.

// Standard pose connection lines (pair of landmark indices)
const POSE_CONNECTIONS = [
  [11, 12], // Shoulders
  [11, 23], // Left side (torso)
  [12, 24], // Right side (torso)
  [23, 24], // Hips
  [11, 13], [13, 15], // Left arm
  [12, 14], [14, 16], // Right arm
  [23, 25], [25, 27], // Left leg
  [24, 26], [26, 28]  // Right leg
];

// Highlight face landmark categories for visual feedback (subset of 478)
// Indices for key facial features (based on standard MediaPipe FaceMesh mapping)
const FACE_CONTOUR = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
  400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
  54, 103, 67, 109
];
const LIPS = [
  78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, // Lower lip outer
  308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78,   // Upper lip outer
  78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,   // Upper lip inner
  308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78    // Lower lip inner
];
const L_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466];
const R_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
const L_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46];
const R_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276];

export class Visualizer {
  static draw(canvas, faceResult, poseResult, defaultEmotionColor = "#8B5CF6", isMirrored = false, primaryEmotionResult = null) {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1. Draw Pose Landmarker skeleton
    if (poseResult && poseResult.landmarks && poseResult.landmarks.length > 0) {
      this.drawPose(ctx, poseResult.landmarks[0], canvas.width, canvas.height, isMirrored);
    }

    // 2. Draw ALL detected faces, bounding boxes, and emotion labels
    if (faceResult && faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0) {
      for (let i = 0; i < faceResult.faceLandmarks.length; i++) {
        const landmarks = faceResult.faceLandmarks[i];
        
        // Classify emotion for this specific face index
        // Use primaryEmotionResult for the primary face (index 0) if available
        const emotionRes = (i === 0 && primaryEmotionResult) ? primaryEmotionResult : classifyEmotion(faceResult, i);
        const metadata = EMOTION_METADATA[emotionRes.dominantEmotion];
        const emotionColor = metadata ? metadata.color : defaultEmotionColor;
        
        // Draw face mesh coordinates (point cloud and contours)
        this.drawFace(ctx, landmarks, canvas.width, canvas.height, emotionColor, isMirrored);

        // Draw bounding box and label above the face
        this.drawFaceBoundingBox(ctx, landmarks, canvas.width, canvas.height, emotionRes, metadata, emotionColor, isMirrored);
      }
    }
  }

  static drawFaceBoundingBox(ctx, landmarks, width, height, emotionRes, metadata, color, isMirrored) {
    let minX = 1, maxX = 0, minY = 1, maxY = 0;
    for (const lm of landmarks) {
      if (lm.x < minX) minX = lm.x;
      if (lm.x > maxX) maxX = lm.x;
      if (lm.y < minY) minY = lm.y;
      if (lm.y > maxY) maxY = lm.y;
    }

    // Account for mirroring if isMirrored is true
    let x = isMirrored ? (1.0 - maxX) * width : minX * width;
    let y = minY * height;
    let w = (maxX - minX) * width;
    let h = (maxY - minY) * height;

    // Add padding (12%)
    const paddingX = w * 0.12;
    const paddingY = h * 0.12;
    x -= paddingX;
    y -= paddingY;
    w += paddingX * 2;
    h += paddingY * 2;

    // Clamp coordinates to canvas bounds
    x = Math.max(0, x);
    y = Math.max(0, y);
    w = Math.min(width - x, w);
    h = Math.min(height - y, h);

    // Draw glowing border box
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.shadowBlur = 8;
    ctx.shadowColor = color;
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Draw label bar above the box
    const labelHeight = 24;
    let labelY = y - labelHeight;
    if (labelY < 0) {
      labelY = y;
    }

    ctx.fillStyle = color;
    ctx.fillRect(x, labelY, w, labelHeight);

    // Text details
    ctx.fillStyle = "#FFFFFF";
    ctx.font = "bold 12px 'Plus Jakarta Sans', sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    
    const percentage = Math.round(emotionRes.confidence * 100);
    const labelText = `${metadata.emoji} ${metadata.label} (${percentage}%)`;
    ctx.fillText(labelText, x + 8, labelY + labelHeight / 2);
  }

  static drawPose(ctx, landmarks, width, height, isMirrored) {
    const threshold = 0.45;

    // Helper to map normalized coordinates to canvas space, mirroring X if requested
    const getPos = (lm) => ({
      x: isMirrored ? (1.0 - lm.x) * width : lm.x * width,
      y: lm.y * height,
      visible: (lm.visibility || 0) > threshold
    });

    // Draw connection lines
    ctx.lineWidth = 4;
    ctx.lineCap = "round";
    ctx.shadowBlur = 10;
    ctx.shadowColor = "rgba(6, 182, 212, 0.6)"; // Cyan glow

    for (const [i1, i2] of POSE_CONNECTIONS) {
      const lm1 = landmarks[i1];
      const lm2 = landmarks[i2];

      if (lm1 && lm2) {
        const pt1 = getPos(lm1);
        const pt2 = getPos(lm2);

        if (pt1.visible && pt2.visible) {
          // Draw skeleton lines in a neon blue-cyan gradient
          const grad = ctx.createLinearGradient(pt1.x, pt1.y, pt2.x, pt2.y);
          grad.addColorStop(0, "rgba(6, 182, 212, 0.85)"); // Cyan
          grad.addColorStop(1, "rgba(59, 130, 246, 0.85)"); // Blue
          
          ctx.strokeStyle = grad;
          ctx.beginPath();
          ctx.moveTo(pt1.x, pt1.y);
          ctx.lineTo(pt2.x, pt2.y);
          ctx.stroke();
        }
      }
    }

    // Draw joints as glowing nodes
    ctx.shadowBlur = 15;
    ctx.shadowColor = "#FFFFFF";

    for (let i = 11; i <= 28; i++) {
      const lm = landmarks[i];
      if (lm) {
        const pt = getPos(lm);
        if (pt.visible) {
          // Circle center
          ctx.fillStyle = "#FFFFFF";
          ctx.strokeStyle = "#0891B2"; // Dark cyan border
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, 6, 0, 2 * Math.PI);
          ctx.fill();
          ctx.stroke();
        }
      }
    }

    // Reset shadow
    ctx.shadowBlur = 0;
  }

  static drawFace(ctx, landmarks, width, height, emotionColor, isMirrored) {
    const getPos = (lm) => ({
      x: isMirrored ? (1.0 - lm.x) * width : lm.x * width,
      y: lm.y * height
    });

    // Draw entire face landmark points as a delicate, transparent violet point cloud
    ctx.fillStyle = "rgba(224, 242, 254, 0.25)"; // Translucent light blue
    for (let i = 0; i < landmarks.length; i++) {
      const pt = getPos(landmarks[i]);
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 1.2, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Draw outlines for eyes, eyebrows, and lips in the dominant emotion color
    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";
    ctx.shadowBlur = 5;
    ctx.shadowColor = emotionColor;
    ctx.strokeStyle = emotionColor;

    const drawPath = (indices, close = false) => {
      if (indices.length === 0) return;
      ctx.beginPath();
      const first = getPos(landmarks[indices[0]]);
      ctx.moveTo(first.x, first.y);
      for (let i = 1; i < indices.length; i++) {
        const pt = getPos(landmarks[indices[i]]);
        ctx.lineTo(pt.x, pt.y);
      }
      if (close) {
        ctx.closePath();
      }
      ctx.stroke();
    };

    drawPath(FACE_CONTOUR, false);
    drawPath(LIPS, true);
    drawPath(L_EYE, true);
    drawPath(R_EYE, true);
    drawPath(L_BROW, false);
    drawPath(R_BROW, false);

    // Draw eyes pupils slightly brighter
    ctx.fillStyle = emotionColor;
    ctx.shadowBlur = 8;
    ctx.shadowColor = "#FFFFFF";

    // MediaPipe key pupil indices
    const lPupilIdx = 468;
    const rPupilIdx = 473;
    if (landmarks[lPupilIdx]) {
      const pt = getPos(landmarks[lPupilIdx]);
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 2.5, 0, 2 * Math.PI);
      ctx.fill();
    }
    if (landmarks[rPupilIdx]) {
      const pt = getPos(landmarks[rPupilIdx]);
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 2.5, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Reset shadow
    ctx.shadowBlur = 0;
  }
}
