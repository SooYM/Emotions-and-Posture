export const POSTURES = {
  STANDING: "standing",
  SITTING: "sitting",
  LYING_DOWN: "lying_down",
  UNKNOWN: "unknown"
};

export const POSTURE_METADATA = {
  [POSTURES.STANDING]: { label: "Standing", icon: "🧍", color: "#10B981" }, // Emerald green
  [POSTURES.SITTING]: { label: "Sitting", icon: "🪑", color: "#3B82F6" },     // Blue
  [POSTURES.LYING_DOWN]: { label: "Lying Down", icon: "🛏️", color: "#F59E0B" }, // Amber
  [POSTURES.UNKNOWN]: { label: "Unknown (Partial Body)", icon: "👤", color: "#6B7280" } // Gray
};

// Calculate 2D angle between vectors BA and BC (vertex B)
function calculateAngle2D(a, b, c) {
  const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let angle = Math.abs((radians * 180.0) / Math.PI);
  if (angle > 180.0) {
    angle = 360.0 - angle;
  }
  return angle;
}

export function classifyPosture(poseResult) {
  // Default values when no pose is detected
  if (!poseResult || !poseResult.landmarks || poseResult.landmarks.length === 0) {
    return {
      posture: POSTURES.UNKNOWN,
      confidence: 0,
      details: "No body detected. Make sure you are in frame.",
      angles: { hipAngle: null, kneeAngle: null, trunkAngle: null }
    };
  }

  const landmarks = poseResult.landmarks[0];
  const threshold = 0.20; // Lowered from 0.45 to support half-body poses using MediaPipe's occluded joint predictions

  // Keypoint indices:
  // Shoulders: Left 11, Right 12
  // Hips: Left 23, Right 24
  // Knees: Left 25, Right 26
  // Ankles: Left 27, Right 28

  const lShoulder = landmarks[11];
  const rShoulder = landmarks[12];
  const lHip = landmarks[23];
  const rHip = landmarks[24];
  const lKnee = landmarks[25];
  const rKnee = landmarks[26];
  const lAnkle = landmarks[27];
  const rAnkle = landmarks[28];

  // Check visibility of major groups
  const lShoulderVisible = lShoulder && (lShoulder.visibility || 0) > threshold;
  const rShoulderVisible = rShoulder && (rShoulder.visibility || 0) > threshold;
  const lHipVisible = lHip && (lHip.visibility || 0) > threshold;
  const rHipVisible = rHip && (rHip.visibility || 0) > threshold;
  const lKneeVisible = lKnee && (lKnee.visibility || 0) > threshold;
  const rKneeVisible = rKnee && (rKnee.visibility || 0) > threshold;
  const lAnkleVisible = lAnkle && (lAnkle.visibility || 0) > threshold;
  const rAnkleVisible = rAnkle && (rAnkle.visibility || 0) > threshold;

  const hipsVisible = lHipVisible || rHipVisible;
  const kneesVisible = lKneeVisible || rKneeVisible;
  const shouldersVisible = lShoulderVisible || rShoulderVisible;
  const anklesVisible = lAnkleVisible || rAnkleVisible;

  // We need at least Shoulders + Hips + Knees to make any posture decision.
  // If not visible, classify as UNKNOWN (Upper body only)
  if (!shouldersVisible || !hipsVisible || !kneesVisible) {
    return {
      posture: POSTURES.UNKNOWN,
      confidence: 0.2,
      details: "Upper body or face only detected. Step back to show hips and knees.",
      angles: { hipAngle: null, kneeAngle: null, trunkAngle: null }
    };
  }

  // Determine side with better visibility to calculate angles
  const lVisibility = (lShoulder.visibility || 0) + (lHip.visibility || 0) + (lKnee.visibility || 0) + (lAnkle ? (lAnkle.visibility || 0) : 0);
  const rVisibility = (rShoulder.visibility || 0) + (rHip.visibility || 0) + (rKnee.visibility || 0) + (rAnkle ? (rAnkle.visibility || 0) : 0);
  
  const useLeft = lVisibility >= rVisibility;
  const shoulder = useLeft ? lShoulder : rShoulder;
  const hip = useLeft ? lHip : rHip;
  const knee = useLeft ? lKnee : rKnee;
  const ankle = useLeft ? lAnkle : rAnkle;
  const isAnkleVisible = useLeft ? lAnkleVisible : rAnkleVisible;

  // 1. Calculate angles
  // Hip angle: Shoulder-Hip-Knee
  const hipAngle = calculateAngle2D(shoulder, hip, knee);
  // Knee angle: Hip-Knee-Ankle (if ankle is visible)
  const kneeAngle = isAnkleVisible ? calculateAngle2D(hip, knee, ankle) : null;

  // Trunk angle relative to horizontal: 0 is horizontal, 90 is vertical
  const dy = hip.y - shoulder.y;
  const dx = hip.x - shoulder.x;
  const trunkAngle = Math.abs(Math.atan2(dy, dx) * 180 / Math.PI);

  // 2. Classify posture
  let posture = POSTURES.UNKNOWN;
  let details = "";
  let confidence = 0.5;

  // Check Lying Down first (trunk is highly horizontal)
  // Normally vertical standing has trunk angle around 70 to 110 degrees.
  // Horizontal has trunk angle close to 0, 180, or < 35 degrees.
  if (trunkAngle < 35 || trunkAngle > 145) {
    posture = POSTURES.LYING_DOWN;
    details = `Lying Down (Horizontal body line: Trunk angle is ${Math.round(trunkAngle)}°)`;
    confidence = 0.85;
  } else {
    // If not lying down, evaluate hip and knee angles
    if (isAnkleVisible) {
      if (hipAngle > 150 && kneeAngle > 150) {
        posture = POSTURES.STANDING;
        details = `Standing (Straight legs: Hip angle ${Math.round(hipAngle)}°, Knee angle ${Math.round(kneeAngle)}°)`;
        confidence = 0.95;
      } else if (hipAngle < 135 && kneeAngle < 135) {
        posture = POSTURES.SITTING;
        details = `Sitting (Bent joints: Hip angle ${Math.round(hipAngle)}°, Knee angle ${Math.round(kneeAngle)}°)`;
        confidence = 0.95;
      } else if (hipAngle < 135 && kneeAngle > 145) {
        // Hip bent but knees straight (e.g., sitting on floor with legs extended)
        posture = POSTURES.SITTING;
        details = `Sitting (Legs extended: Hip angle ${Math.round(hipAngle)}°, Knee angle ${Math.round(kneeAngle)}°)`;
        confidence = 0.8;
      } else if (hipAngle > 145 && kneeAngle < 130) {
        // Hip straight, knees bent (e.g., kneeling, crouching, or squatting)
        posture = POSTURES.STANDING;
        details = `Crouching/Squatting (Knee angle ${Math.round(kneeAngle)}°)`;
        confidence = 0.7;
      } else {
        // Intermediate/uncertain
        posture = POSTURES.UNKNOWN;
        details = `Transitioning / Active posture (Hip angle ${Math.round(hipAngle)}°, Knee angle ${Math.round(kneeAngle)}°)`;
        confidence = 0.6;
      }
    } else {
      // Fallback: Ankles are NOT visible, make decision based on Hip Angle only
      if (hipAngle > 150) {
        posture = POSTURES.STANDING;
        details = `Standing (Hips straight: Hip angle ${Math.round(hipAngle)}°, ankles blocked)`;
        confidence = 0.75;
      } else if (hipAngle < 130) {
        posture = POSTURES.SITTING;
        details = `Sitting (Hips bent: Hip angle ${Math.round(hipAngle)}°, ankles blocked)`;
        confidence = 0.8;
      } else {
        posture = POSTURES.UNKNOWN;
        details = `Unsure (Hip angle is intermediate: ${Math.round(hipAngle)}° and legs hidden)`;
        confidence = 0.5;
      }
    }
  }

  return {
    posture,
    confidence,
    details,
    angles: {
      hipAngle: Math.round(hipAngle),
      kneeAngle: kneeAngle !== null ? Math.round(kneeAngle) : null,
      trunkAngle: Math.round(trunkAngle)
    }
  };
}
