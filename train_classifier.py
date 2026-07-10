import os
import urllib.request
import cv2
import numpy as np
import json
import pickle
import time
import argparse
import warnings
from datetime import datetime
import sys
# Suppress stderr during mediapipe and protobuf imports to prevent noisy "GetPrototype" warnings
_original_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except Exception:
    sys.stderr = _original_stderr
    raise
finally:
    sys.stderr = _original_stderr
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Ignore false system-level vecLib/BLAS matmul runtime warnings on macOS
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configuration
DATASET_DIR = "Dataset"
CACHE_PATH = "dataset_features.npz"
MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
OUTPUT_JSON = "public/model_weights.json"
SAVED_MODEL_PKL = "emotion_model.pkl"
LOG_FILE = "training_log.txt"

EMOTIONS = ["happy", "sad", "anxiety", "angry", "surprised", "disgust", "neutral"]

FOLDER_TO_EMOTION = {
    "happy": "happy",
    "sad": "sad",
    "fear": "anxiety",
    "anxiety": "anxiety",
    "angry": "angry",
    "surprise": "surprised",
    "surprised": "surprised",
    "disgust": "disgust",
    "neutral": "neutral"
}

def download_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading Face Landmarker model task to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully.")

def extract_features():
    # Setup Detector with GPU delegation
    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH,
        delegate=python.BaseOptions.Delegate.GPU
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    
    feature_names = []
    
    def process_split(split_name):
        split_dir = os.path.join(DATASET_DIR, split_name)
        if not os.path.exists(split_dir):
            print(f"Directory {split_dir} not found. Skipping split.")
            return np.array([]), np.array([]), []
            
        X = []
        y = []
        paths = []
        
        print(f"Processing split: {split_name}...")
        folders = [f for f in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, f))]
        
        total_files = 0
        processed_files = 0
        faces_detected = 0
        
        for folder in folders:
            emotion_key = FOLDER_TO_EMOTION.get(folder.lower())
            if not emotion_key:
                continue
            folder_path = os.path.join(split_dir, folder)
            total_files += len([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
            
        print(f"Found {total_files} images in split '{split_name}'. Starting MediaPipe extraction...")
        
        for folder in folders:
            emotion_key = FOLDER_TO_EMOTION.get(folder.lower())
            if not emotion_key:
                continue
                
            label_idx = EMOTIONS.index(emotion_key)
            folder_path = os.path.join(split_dir, folder)
            
            for file_name in os.listdir(folder_path):
                if not file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    continue
                    
                processed_files += 1
                if processed_files % 50 == 0 or processed_files == total_files:
                    print(f"\rProgress: {processed_files}/{total_files} images processed | Faces detected: {faces_detected}", end="", flush=True)
                    
                file_path = os.path.join(folder_path, file_name)
                
                try:
                    img = cv2.imread(file_path)
                    if img is None:
                        continue
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                    result = detector.detect(mp_image)
                    
                    if result.face_blendshapes and len(result.face_blendshapes) > 0:
                        blendshapes = result.face_blendshapes[0]
                        feature_values = []
                        current_names = []
                        
                        for item in blendshapes:
                            feature_values.append(item.score)
                            current_names.append(item.category_name)
                            
                        nonlocal feature_names
                        if not feature_names:
                            feature_names = current_names
                            print(f"Detected {len(feature_names)} blendshape features.")
                            
                        X.append(feature_values)
                        y.append(label_idx)
                        paths.append(os.path.join(split_name, folder, file_name))
                        faces_detected += 1
                except Exception as e:
                    print(f"Error processing image {file_path}: {e}")
                    
        print() # Move to new line after carriage return progress updates
        print(f"Finished split '{split_name}'. Total: {processed_files}, Face Detected: {faces_detected} ({int(faces_detected/max(1, processed_files)*100)}%)")
        return np.array(X), np.array(y), paths

    X_train, y_train, paths_train = process_split("train")
    X_test, y_test, paths_test = process_split("test")
    
    print("Saving extracted features cache...")
    np.savez_compressed(
        CACHE_PATH,
        X_train=X_train, y_train=y_train, paths_train=paths_train,
        X_test=X_test, y_test=y_test, paths_test=paths_test,
        feature_names=feature_names
    )
    print(f"Saved cache to {CACHE_PATH}.")
    return X_train, y_train, X_test, y_test, feature_names

def main():
    parser = argparse.ArgumentParser(description="Train MLP Emotion Classifier with incremental warm start.")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="Number of epochs to train in this session.")
    args = parser.parse_args()
    
    download_model()
    
    # Load features from Cache or run extraction
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached features from {CACHE_PATH}...")
        cache = np.load(CACHE_PATH, allow_pickle=True)
        X_train = cache["X_train"]
        y_train = cache["y_train"]
        X_test = cache["X_test"]
        y_test = cache["y_test"]
        feature_names = list(cache["feature_names"])
        print(f"Loaded train: {X_train.shape}, test: {X_test.shape}")
    else:
        print("No cache found. Starting full feature extraction...")
        X_train, y_train, X_test, y_test, feature_names = extract_features()
        
    if len(X_train) == 0 or len(X_test) == 0:
        print("Insufficient training/test data to train model.")
        return

    # Load or initialize model
    if os.path.exists(SAVED_MODEL_PKL):
        print(f"Loading existing model from checkpoint: {SAVED_MODEL_PKL}...")
        with open(SAVED_MODEL_PKL, "rb") as f:
            clf = pickle.load(f)
        print(f"Resuming model training. Total epochs trained so far: {getattr(clf, 'total_epochs_', 0)}")
    else:
        print("Initializing new Multi-Layer Perceptron (MLP) Classifier...")
        # 52 inputs -> 128 hidden neurons -> 64 hidden neurons -> 7 outputs
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.01,
            learning_rate_init=0.001,
            warm_start=False,
            random_state=42
        )
        clf.total_epochs_ = 0

    classes = np.array(range(len(EMOTIONS)))
    epochs_to_train = args.epochs
    
    log_entries = []
    start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entries.append(f"\n==========================================")
    log_entries.append(f"TRAINING SESSION STARTED AT: {start_time_str}")
    log_entries.append(f"Requested Epochs in this Session: {epochs_to_train}")
    log_entries.append(f"Total Epochs Before Session: {clf.total_epochs_}")
    log_entries.append(f"==========================================")
    
    print("\n" + log_entries[-4])
    print(log_entries[-3])
    print(log_entries[-2])
    print(log_entries[-1] + "\n")
    
    session_start_time = time.time()
    
    for epoch in range(1, epochs_to_train + 1):
        epoch_start_time = time.time()
        
        # Shuffle training indices for high-quality mini-batch convergence
        indices = np.arange(len(X_train))
        np.random.seed(42 + clf.total_epochs_)
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        # Train in mini-batches of size 256
        batch_size = 256
        for i in range(0, len(X_train_shuffled), batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            clf.partial_fit(X_batch, y_batch, classes=classes)
            
        # Decay learning rate slightly per epoch to help gradient descent settle smoothly
        clf.learning_rate_init = max(0.0001, clf.learning_rate_init * 0.98)
            
        clf.total_epochs_ += 1
        
        epoch_duration = time.time() - epoch_start_time
        
        # Evaluate performance on train and test sets
        loss = clf.loss_
        y_pred_train = clf.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train)
        
        y_pred_test = clf.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        log_line = (f"Epoch {clf.total_epochs_} (Session Epoch {epoch}/{epochs_to_train}) | "
                    f"Loss: {loss:.4f} | "
                    f"Train Acc: {train_acc*100:.2f}% | "
                    f"Test Acc: {test_acc*100:.2f}% | "
                    f"Time: {epoch_duration:.2f}s")
        print(log_line, flush=True)
        log_entries.append(log_line)

    session_duration = time.time() - session_start_time
    
    # Classification report on final test set
    y_pred_test_final = clf.predict(X_test)
    final_acc = accuracy_score(y_test, y_pred_test_final)
    
    summary_line = (f"==========================================\n"
                    f"SESSION FINISHED AT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Session Duration: {session_duration:.2f}s\n"
                    f"Total Cumulative Epochs: {clf.total_epochs_}\n"
                    f"Final Test Set Accuracy: {final_acc*100:.2f}%\n"
                    f"==========================================")
    print("\n" + summary_line)
    log_entries.append(summary_line)
    
    # Save checkpoint model
    with open(SAVED_MODEL_PKL, "wb") as f:
        pickle.dump(clf, f)
    print(f"Saved model checkpoint to {SAVED_MODEL_PKL}.")

    # Append to log file
    with open(LOG_FILE, "a") as f:
        f.write("\n".join(log_entries) + "\n")
    print(f"Logged training session metrics to {LOG_FILE}.")

    # Export weights for JavaScript browser inference
    # MLP coefs_[0] shape is (52, 128), coefs_[1] shape is (128, 64), coefs_[2] shape is (64, 7)
    # intercepts_[0] shape is (128,), intercepts_[1] shape is (64,), intercepts_[2] shape is (7,)
    model_weights = {
        "features": feature_names,
        "classes": EMOTIONS,
        "type": "mlp",
        "w1": clf.coefs_[0].tolist(), # shape (52, 128)
        "b1": clf.intercepts_[0].tolist(), # shape (128,)
        "w2": clf.coefs_[1].tolist(), # shape (128, 64)
        "b2": clf.intercepts_[1].tolist(), # shape (64,)
        "w3": clf.coefs_[2].tolist(), # shape (64, 7)
        "b3": clf.intercepts_[2].tolist() # shape (7,)
    }
    
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(model_weights, f, indent=2)
    print(f"Successfully exported MLP weights to Web App: {OUTPUT_JSON}.\n")

if __name__ == "__main__":
    main()
