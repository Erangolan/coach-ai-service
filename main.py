from fastapi import FastAPI, UploadFile, File, Form, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import uuid
import os
import shutil
import cv2
import mediapipe as mp
import math
import json
from fastapi import HTTPException
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
import base64
import io
from typing import List, Dict, Any
from collections import deque

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typing import List
import time
import logging
from datetime import datetime
from pose_utils import LSTMClassifier, extract_angles, extract_sequence_from_video, CNN_LSTM_Classifier, LSTM_Transformer_Classifier, LSTM_GNN_Classifier, extract_keypoints_xyz, get_angle_indices_by_parts, add_velocity_features

# Setup logging
def setup_prediction_logger(exercise_name):
    """Setup logger for prediction tracking"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(f'prediction_{exercise_name}')
    logger.setLevel(logging.INFO)
    
    # Create file handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'logs/predictions_{exercise_name}_{timestamp}.log'
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

DATABASE_URL = "postgresql://erangolan:eran1234@localhost:5432/exercise_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Important: Same label mapping as in training
LABELS_MAP = {
    "good": 0,
    "bad-left-angle": 1,
    "bad-lower-knee": 2,
    "bad-right-angle": 3,
    "idle": 4,
}
LABELS = list(LABELS_MAP.keys())
label_to_idx = LABELS_MAP

def load_model(exercise_name, model_type='cnn_lstm', input_size=40, num_classes=5, bidirectional=True):
    model_path = f"models/{exercise_name}_model.pt"
    if not os.path.exists(model_path):
        raise ValueError(f"Model not found: {model_path}")
    if model_type == 'cnn_lstm':
        model = CNN_LSTM_Classifier(input_size=input_size, num_classes=num_classes, bidirectional=bidirectional)
    elif model_type == 'lstm_transformer':
        model = LSTM_Transformer_Classifier(input_size=input_size, num_classes=num_classes, bidirectional=bidirectional)

    else:
        model = LSTMClassifier(input_size=input_size, num_classes=num_classes, bidirectional=bidirectional)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def preprocess_video_to_sequence(video_path: str) -> torch.Tensor:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    timestamp = 0.0
    seq = []

    while timestamp < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        angles = extract_angles(frame)
        if angles:
            values = [angles.get(joint, 0.0) for joint in [
                'knee', 'hip', 'elbow', 'shoulder', 'ankle',
                'wrist', 'neck', 'back', 'ankle_dorsiflexion'
            ]]
            seq.append(values)
        timestamp += 0.1

    cap.release()
    return torch.tensor(seq, dtype=torch.float32)

# DB MODELS
class Exercise(Base):
    __tablename__ = "exercises"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)

class PoseFrame(Base):
    __tablename__ = "pose_frames"
    id = Column(Integer, primary_key=True, index=True)
    exercise_id = Column(Integer, ForeignKey("exercises.id"))
    timestamp = Column(Float)
    angles = Column(JSON)  # {'knee': 145.0, 'hip': 90.0, ...}

Base.metadata.create_all(bind=engine)

# UTILS

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def calculate_angle(a, b, c):
    def vector(p1, p2):
        return [p2[0] - p1[0], p2[1] - p1[1]]
    ba = vector(b, a)
    bc = vector(b, c)
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    norm_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    norm_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    cosine_angle = dot / (norm_ba * norm_bc + 1e-8)
    angle = math.degrees(math.acos(min(1, max(-1, cosine_angle))))
    return angle

def compute_dtw_distance(seq1, seq2):
    """
    השוואת שתי סדרות של זוויות מפרקים
    seq1, seq2 = [ {knee: x, hip: y, ...}, ... ]
    """
    if not seq1 or not seq2:
        return 0.0

    keys = list(seq1[0].keys())

    def to_vector(seq):
        return [np.array([frame[k] for k in keys if k in frame]) for frame in seq]

    v1 = to_vector(seq1)
    v2 = to_vector(seq2)

    distance, _ = fastdtw(v1, v2, dist=euclidean)
    normalized = distance / max(len(v1), len(v2))
    similarity = max(0.0, 100 - normalized)  # 100 = הכי דומה
    return round(similarity, 2)

@app.post("/upload_reference/")
def upload_reference(exercise_name: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    if db.query(Exercise).filter_by(name=exercise_name).first():
        raise HTTPException(status_code=400, detail="Exercise already exists")

    video_path = f"/tmp/{uuid.uuid4()}.mp4"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    exercise = Exercise(name=exercise_name)
    db.add(exercise)
    db.commit()
    db.refresh(exercise)

    timestamp = 0.0
    while timestamp < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        angles = extract_angles(frame)
        if angles is None:
            timestamp += 0.1
            continue
        pose = PoseFrame(
            timestamp=round(timestamp, 2),
            angles=json.dumps(angles),
            exercise_id=exercise.id
        )
        db.add(pose)
        print(f"Saved frame at {timestamp:.2f}s")
        timestamp += 0.1

    db.commit()
    cap.release()
    os.remove(video_path)

    return {"message": "Reference saved", "exercise_id": exercise.id}

@app.post("/evaluate_exercise/")
def evaluate_exercise(exercise_name: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    exercise = db.query(Exercise).filter_by(name=exercise_name).first()
    if not exercise:
        raise HTTPException(status_code=404, detail="Reference exercise not found")

    # Load reference frames
    reference_frames = db.query(PoseFrame).filter_by(exercise_id=exercise.id).order_by(PoseFrame.timestamp).all()
    ref_sequence = [json.loads(frame.angles) for frame in reference_frames]

    # Save video
    video_path = f"/tmp/{uuid.uuid4()}.mp4"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract angles from new video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    new_sequence = []
    timestamp = 0.0
    while timestamp < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        angles = extract_angles(frame)
        if angles:
            new_sequence.append(angles)
        timestamp += 0.1

    cap.release()
    os.remove(video_path)

    if not new_sequence:
        raise HTTPException(status_code=400, detail="No valid pose data found in uploaded video")

    # Compute similarity
    score = compute_dtw_distance(ref_sequence, new_sequence)

    return {"similarity_score": score, "message": f"Similarity to reference: {score}%"}

@app.post("/classify/")
def classify_video(file: UploadFile = File(...), exercise_name: str = Form(...)):
    # 1. שמירת הקובץ
    video_path = f"/tmp/{uuid.uuid4()}.mp4"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. עיבוד
    sequence = preprocess_video_to_sequence(video_path)
    os.remove(video_path)

    if len(sequence) == 0:
        raise HTTPException(status_code=400, detail="No valid pose detected")

    # 3. חיזוי
    focus_parts = ['right_knee', 'right_shoulder', 'torso']
    use_keypoints = True
    use_velocity = True
    use_statistics = True
    use_ratios = True
    focus_indices = get_angle_indices_by_parts(focus_parts)
    base_features = len(focus_indices) + (12 * 3 if use_keypoints else 0)
    input_size = base_features
    if use_velocity:
        input_size *= 3
    if use_statistics:
        input_size += input_size * 6
    if use_ratios:
        input_size += 1
    
    model = load_model(exercise_name, model_type='cnn_lstm', input_size=input_size, num_classes=5, bidirectional=True)
    input_tensor = sequence.unsqueeze(0)  # [1, seq_len, 9]
    lengths = torch.tensor([sequence.shape[0]])
    output = model(input_tensor, lengths)
    predicted = torch.argmax(output, dim=1).item()
    predicted_label = LABELS[predicted]

    return {"prediction": predicted_label}

@app.post("/analyze/")
async def analyze_video(file: UploadFile = File(...), exercise_name: str = Form("right-knee-to-90-degrees")):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")
        
    if not file.filename.endswith('.mp4'):
        raise HTTPException(status_code=400, detail="Only MP4 files are supported")
    
    # Save the file
    filename = file.filename
    video_path = os.path.join('uploads', filename)
    os.makedirs('uploads', exist_ok=True)
    
    with open(video_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        # Extract angle sequence
        sequence = extract_sequence_from_video(video_path)
        if not sequence:
            raise HTTPException(status_code=400, detail="No valid poses detected in video")
        
        # Load model
        focus_parts = ['right_knee', 'right_shoulder', 'torso']
        use_keypoints = True
        use_velocity = True
        use_statistics = True  # Enable statistical features
        use_ratios = True
        focus_indices = get_angle_indices_by_parts(focus_parts)
        base_features = len(focus_indices) + (12 * 3 if use_keypoints else 0)
        input_size = base_features
        if use_velocity:
            input_size *= 3  # Triple the input size for velocity + acceleration features
        if use_statistics:
            input_size += input_size * 6  # Add 6 statistical features per original feature (mean, median, std, max, min, range)
        if use_ratios:
            input_size += 1
        
        model = load_model(exercise_name, model_type='cnn_lstm', input_size=input_size, num_classes=5, bidirectional=True)
        
        # Prepare data for model
        sequence_tensor = torch.tensor([sequence], dtype=torch.float32)
        lengths = torch.tensor([len(sequence)])
        
        # Prediction
        with torch.no_grad():
            outputs = model(sequence_tensor, lengths)
            _, predicted = torch.max(outputs, 1)
            
        # Cleanup
        os.remove(video_path)
        
        # Return actual label from model instead of binary good/bad
        predicted_label = LABELS[predicted.item()]
        
        return {
            'prediction': predicted_label,
            'confidence': torch.softmax(outputs, dim=1)[0][predicted].item()
        }
        
    except Exception as e:
        if os.path.exists(video_path):
            os.remove(video_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_exercise/")
async def predict_exercise(file: UploadFile = File(...), exercise_name: str = Form(...)):
    """Predict if the movement in the uploaded video is good or bad for the given exercise."""
    import uuid
    import os
    import torch
    from fastapi import HTTPException
    from pose_utils import extract_sequence_from_video

    # Save uploaded video to a temp file
    video_path = f"/tmp/{uuid.uuid4()}.mp4"
    with open(video_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # === בדיוק כמו באימון! ===
        focus_parts = ['right_knee', 'right_shoulder', 'torso']
        use_keypoints = True
        use_velocity = True
        use_statistics = True  # Enable statistical features
        use_ratios = True
        focus_indices = get_angle_indices_by_parts(focus_parts)
        base_features = len(focus_indices) + (12 * 3 if use_keypoints else 0)
        input_size = base_features
        if use_velocity:
            input_size *= 3  # Triple the input size for velocity + acceleration features
        if use_statistics:
            input_size += input_size * 6  # Add 6 statistical features per original feature (mean, median, std, max, min, range)
        if use_ratios:
            input_size += 1
        
        # Extract sequence from video using the correct pipeline
        sequence = extract_sequence_from_video(video_path, focus_indices=focus_indices, use_keypoints=use_keypoints, use_velocity=use_velocity, use_statistics=use_statistics, use_ratios=use_ratios)
        if sequence is None or len(sequence) == 0:
            raise HTTPException(status_code=400, detail="No valid pose detected in video.")
        
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)

        # Load model for the given exercise
        try:
            model = load_model(exercise_name, model_type='cnn_lstm', input_size=input_size, num_classes=5, bidirectional=True)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Model for exercise '{exercise_name}' not found: {str(e)}")

        # Prepare input for model
        input_tensor = sequence_tensor.unsqueeze(0)  # [1, seq_len, features]
        lengths = torch.tensor([sequence_tensor.shape[0]])
        with torch.no_grad():
            output = model(input_tensor, lengths)
            predicted = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1)[0][predicted].item()
        
        # Return actual label from model instead of binary good/bad
        predicted_label = LABELS[predicted]
        return {"prediction": predicted_label, "confidence": confidence}
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

@app.websocket("/ws/video-analysis-base64/{exercise_name}")
async def websocket_video_analysis_base64(websocket: WebSocket, exercise_name: str):
    await websocket.accept()
    holistic = None
    
    # Setup prediction logger
    prediction_logger = setup_prediction_logger(exercise_name)
    prediction_logger.info(f"Starting prediction session for exercise: {exercise_name}")

    try:
        focus_parts = ['right_knee', 'right_shoulder', 'torso']
        use_keypoints = True
        use_velocity = True  # Enable velocity features
        use_statistics = True  # Enable statistical features
        use_ratios = True
        focus_indices = get_angle_indices_by_parts(focus_parts)
        base_features = len(focus_indices) + (12 * 3 if use_keypoints else 0)
        input_size = base_features
        if use_velocity:
            input_size *= 3  # Triple the input size for velocity + acceleration features
        if use_statistics:
            input_size += input_size * 6  # Add 6 statistical features per original feature (mean, median, std, max, min, range)
        if use_ratios:
            input_size += 1

        # Debug prints
        print(f"[DEBUG] focus_parts: {focus_parts}")
        print(f"[DEBUG] focus_indices length: {len(focus_indices)}")
        print(f"[DEBUG] Base features (angles + keypoints): {base_features}")
        print(f"[DEBUG] After velocity: {input_size if not use_velocity else 'N/A'}")
        print(f"[DEBUG] After statistics: {input_size if not use_statistics else 'N/A'}")
        print(f"[DEBUG] After ratios: {input_size if not use_ratios else 'N/A'}")
        print(f"[DEBUG] Final input_size: {input_size}")
        print(f"[DEBUG] Expected input_size: 1177")
        print(f"[DEBUG] Match: {input_size == 1177}")

        model = load_model(
            exercise_name, model_type='cnn_lstm',
            input_size=input_size, num_classes=5, bidirectional=True
        )
        model.eval()

        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        LABELS = ["good", "bad-left-angle", "bad-lower-knee", "bad-right-angle", "idle"]
        BUFFER_SIZE = 40  # 1 second at 40 FPS
        ANALYSIS_INTERVAL = 0.2  # seconds
        VOTING_WINDOW_SIZE = 5  # Number of recent predictions to consider
        MIN_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for counting
        MIN_TIME_BETWEEN_REPS = 0.1  # seconds between rep counting
        MIN_AGREEMENT = 3  # At least 3 out of 5 predictions must agree

        # State machine for rep counting
        class RepStateMachine:
            def __init__(self):
                self.state = "idle"  # idle, good, bad-left-angle, bad-lower-knee, bad-right-angle
                self.rep_count = 0
                self.last_state_change_time = 0.0
                self.min_state_duration = 0.1  # Minimum time to stay in a state (seconds)
                self.min_rep_duration = 0.2  # Minimum rep duration (seconds)
                self.max_rep_duration = 2.0  # Maximum rep duration (seconds)
                self.rep_start_time = None
                self.rep_end_time = None
                self.rep_durations = []  # Track rep durations for validation
                self.bad_form_counts = {
                    "bad-left-angle": 0,
                    "bad-lower-knee": 0,
                    "bad-right-angle": 0
                }
                # Track if rep started from good form
                self.rep_started_from_good = False
                # Timeout mechanism
                self.state_timeout = 1.0  # Maximum time to stay in any state (seconds)
            
            def update(self, prediction, confidence, current_time):
                """Update state machine and return if a rep was completed"""
                rep_completed = True
                old_state = self.state
                
                # Check for timeout - if we've been in any state too long, return to idle
                if self.state != "idle" and (current_time - self.last_state_change_time) > self.state_timeout:
                    time_in_state = current_time - self.last_state_change_time
                    prediction_logger.info(f"STATE - Time: {current_time:.2f}s, {old_state} → idle (timeout: {time_in_state:.2f}s > {self.state_timeout}s)")
                    
                    # If we were in good state and have a valid rep, count it
                    if old_state == "good" and self.rep_start_time:
                        rep_duration = current_time - self.rep_start_time
                        if self.min_rep_duration <= rep_duration <= self.max_rep_duration:
                            self.rep_count += 1
                            self.rep_durations.append(rep_duration)
                            rep_completed = True
                            prediction_logger.info(f"STATE - Time: {current_time:.2f}s, {old_state} → idle (rep_completed_timeout) - Duration: {rep_duration:.2f}s - Total reps: {self.rep_count}")
                        else:
                            reason = f"rep_too_short_{rep_duration:.1f}s" if rep_duration < self.min_rep_duration else f"rep_too_long_{rep_duration:.2f}s"
                            prediction_logger.info(f"STATE - Time: {current_time:.2f}s, {old_state} → idle ({reason}_timeout) - Duration: {rep_duration:.2f}s")
                    
                    # Reset to idle state
                    self.state = "idle"
                    self.rep_started_from_good = False
                    self.rep_start_time = None
                    self.last_state_change_time = current_time
                    return rep_completed, "timeout_to_idle"
                
                # Only consider high confidence predictions
                if confidence < MIN_CONFIDENCE_THRESHOLD:
                    prediction_logger.info(f"STATE - Time: {current_time:.2f}s, {old_state} → {old_state} (low_confidence: {confidence:.2f})")
                    return False, "low_confidence"
                
                # Check if enough time has passed since last state change
                if current_time - self.last_state_change_time < self.min_state_duration:
                    time_in_state = current_time - self.last_state_change_time
                    prediction_logger.info(f"STATE - Time: {current_time:.2f}s, {old_state} → {old_state} (state_too_short: {time_in_state:.2f}s < {self.min_state_duration}s)")
                    return False, "state_too_short"
                
                # State transitions
                if self.state == "idle":
                    if prediction == "good":
                        self.state = "good"
                        self.last_state_change_time = current_time
                        self.rep_start_time = current_time
                        self.rep_started_from_good = True  # Mark that rep started from good form
                        prediction_logger.info(f"STATE - Time: {current_time:.2f}s, {old_state} → {self.state} (rep_started) - Prediction: {prediction} (conf: {confidence:.2f}) - rep_started_from_good: {self.rep_started_from_good}")
                        return False, "rep_started"
                    elif prediction in ["bad-left-angle", "bad-lower-knee", "bad-right-angle"]:
                        self.state = prediction  # Use the specific bad form state
                        self.last_state_change_time = current_time
                        self.bad_form_counts[prediction] += 1
                        prediction_logger.info(f"STATE - Time: {current_time:.2f}s, {old_state} → {self.state} (bad_form) - Prediction: {prediction} (conf: {confidence:.2f}) - Count: {self.bad_form_counts[prediction]}")
                        return False, "bad_form"
                
                elif self.state == "good":
                    # Check if we're staying in "good" state
                    if prediction == "good":
                        # We're still in good state, no transition needed
                        prediction_logger.info(f"STATE - Time: {current_time:.2f}s, {old_state} → {old_state} (still_good) - Prediction: {prediction} (conf: {confidence:.2f})")
                        return False, "still_good"
                    
                    # Any transition out of "good" counts as a completed rep
                    if self.rep_start_time:
                        rep_duration = current_time - self.rep_start_time
                        if self.min_rep_duration <= rep_duration <= self.max_rep_duration:
                            self.rep_count += 1
                            self.rep_durations.append(rep_duration)
                            rep_completed = True
                            
                            # Handle the transition based on prediction
                            if prediction == "idle":
                                self.state = "idle"
                                prediction_logger.info(f"STATE - Time: {current_time:.2f}s, {old_state} → {self.state} (rep_completed_to_idle) - Duration: {rep_duration:.2f}s - Total reps: {self.rep_count}")
                            elif prediction in ["bad-left-angle", "bad-lower-knee", "bad-right-angle"]:
                                self.state = prediction
                                self.bad_form_counts[prediction] += 1
                                prediction_logger.info(f"STATE - Time: {current_time:.2f}s, {old_state} → {self.state} (rep_completed_to_bad) - Duration: {rep_duration:.2f}s - Total reps: {self.rep_count} - Bad form: {prediction}")
                            else:
                                # Any other transition
                                self.state = prediction
                                prediction_logger.info(f"STATE - Time: {current_time:.2f}s, {old_state} → {self.state} (rep_completed_to_other) - Duration: {rep_duration:.2f}s - Total reps: {self.rep_count}")
                            
                            # Reset rep tracking
                            self.rep_started_from_good = False
                            self.rep_start_time = None
                            self.last_state_change_time = current_time
                            return rep_completed, f"rep_completed_duration_{rep_duration:.1f}s"
                        else:
                            reason = f"rep_too_short_{rep_duration:.1f}s" if rep_duration < self.min_rep_duration else f"rep_too_long_{rep_duration:.2f}s"
                            
                            # Handle the transition based on prediction
                            if prediction == "idle":
                                self.state = "idle"
                                prediction_logger.info(f"STATE - Time: {current_time:.2f}s, {old_state} → {self.state} ({reason}_to_idle) - Duration: {rep_duration:.2f}s")
                            elif prediction in ["bad-left-angle", "bad-lower-knee", "bad-right-angle"]:
                                self.state = prediction
                                self.bad_form_counts[prediction] += 1
                                prediction_logger.info(f"STATE - Time: {current_time:.2f}s, {old_state} → {self.state} ({reason}_to_bad) - Duration: {rep_duration:.2f}s - Bad form: {prediction}")
                            else:
                                self.state = prediction
                                prediction_logger.info(f"STATE - Time: {current_time:.2f}s, {old_state} → {self.state} ({reason}_to_other) - Duration: {rep_duration:.2f}s")
                            
                            # Reset rep tracking
                            self.rep_started_from_good = False
                            self.rep_start_time = None
                            self.last_state_change_time = current_time
                            return False, reason
                    else:
                        prediction_logger.info(f"STATE - Time: {current_time:.2f}s, {old_state} → {self.state} (no_rep_start_time)")
                        return False, "no_rep_start_time"
                
                elif self.state in ["bad-left-angle", "bad-lower-knee", "bad-right-angle"]:
                    if prediction == "idle":
                        self.state = "idle"
                        self.last_state_change_time = current_time
                        prediction_logger.info(f"STATE - Time: {current_time:.2f}s, {old_state} → {self.state} (returned_to_idle) - Prediction: {prediction} (conf: {confidence:.2f})")
                        return False, "returned_to_idle"
                    elif prediction == "good":
                        self.state = "good"
                        self.last_state_change_time = current_time
                        # Start new rep if not already started from good
                        if not self.rep_started_from_good:
                            self.rep_start_time = current_time
                            self.rep_started_from_good = True
                        prediction_logger.info(f"STATE - Time: {current_time:.2f}s, {old_state} → {self.state} (rep_started_after_bad) - Prediction: {prediction} (conf: {confidence:.2f})")
                        return False, "rep_started_after_bad"
                    elif prediction in ["bad-left-angle", "bad-lower-knee", "bad-right-angle"] and prediction != old_state:
                        # Transition between different bad form states
                        self.state = prediction
                        self.last_state_change_time = current_time
                        self.bad_form_counts[prediction] += 1
                        prediction_logger.info(f"STATE - Time: {current_time:.2f}s, {old_state} → {self.state} (bad_form_changed) - Prediction: {prediction} (conf: {confidence:.2f}) - Count: {self.bad_form_counts[prediction]}")
                        return False, "bad_form_changed"
                
                prediction_logger.info(f"STATE - Time: {current_time:.2f}s, {old_state} → {old_state} (no_change) - Prediction: {prediction} (conf: {confidence:.2f})")
                return False, "no_change"
            
            def get_status_message(self, current_time):
                """Get a human-readable status message"""
                if self.state == "idle":
                    return "Ready for next rep"
                elif self.state == "good":
                    if self.rep_start_time:
                        duration = current_time - self.rep_start_time
                        return f"Good form! ({duration:.1f}s)"
                    return "Good form!"
                elif self.state == "bad-left-angle":
                    return "Fix left angle"
                elif self.state == "bad-lower-knee":
                    return "Lower your knee more"
                elif self.state == "bad-right-angle":
                    return "Fix right angle"
                return "Unknown state"

        frame_buffer = deque(maxlen=BUFFER_SIZE)
        prediction_history = deque(maxlen=VOTING_WINDOW_SIZE)
        rep_counts = {label: 0 for label in LABELS}
        frame_count = 0
        last_analysis_time = 0.0
        current_time = 0.0
        
        # Initialize state machine
        state_machine = RepStateMachine()
        
        # Log initial parameters
        prediction_logger.info(f"INIT - State Machine Parameters:")
        prediction_logger.info(f"INIT -   min_state_duration: {state_machine.min_state_duration}s")
        prediction_logger.info(f"INIT -   min_rep_duration: {state_machine.min_rep_duration}s")
        prediction_logger.info(f"INIT -   max_rep_duration: {state_machine.max_rep_duration}s")
        prediction_logger.info(f"INIT -   state_timeout: {state_machine.state_timeout}s")
        prediction_logger.info(f"INIT -   min_confidence_threshold: {MIN_CONFIDENCE_THRESHOLD}")
        prediction_logger.info(f"INIT -   analysis_interval: {ANALYSIS_INTERVAL}s")
        prediction_logger.info(f"INIT -   voting_window_size: {VOTING_WINDOW_SIZE}")
        prediction_logger.info(f"INIT -   min_agreement: {MIN_AGREEMENT}")

        await websocket.send_json({
            "type": "status",
            "message": f"Ready to analyze {exercise_name} exercise (base64) with state machine rep counting",
            "model_loaded": True,
            "analysis_params": {
                "analysis_interval": ANALYSIS_INTERVAL,
                "voting_window_size": VOTING_WINDOW_SIZE,
                "min_confidence_threshold": MIN_CONFIDENCE_THRESHOLD,
                "min_time_between_reps": MIN_TIME_BETWEEN_REPS,
                "min_agreement": MIN_AGREEMENT,
                "state_machine": {
                    "min_state_duration": state_machine.min_state_duration,
                    "min_rep_duration": state_machine.min_rep_duration,
                    "max_rep_duration": state_machine.max_rep_duration,
                    "state_timeout": state_machine.state_timeout,
                    "states": ["idle", "good", "bad-left-angle", "bad-lower-knee", "bad-right-angle"]
                }
            }
        })

        while True:
            try:
                data = await websocket.receive_text()
                if data.startswith('data:image/'):
                    data = data.split(',')[1]
                binary_data = base64.b64decode(data)
                nparr = np.frombuffer(binary_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)

                if not results.pose_landmarks:
                    continue

                angles = extract_angles(results.pose_landmarks)
                features = [angles[i] for i in focus_indices] if focus_indices else angles
                if use_keypoints:
                    keypoints = extract_keypoints_xyz(results.pose_landmarks)
                    features += keypoints
                frame_buffer.append(features)
                frame_count += 1
                current_time = frame_count / 40.0  # Assuming 40 FPS

                # Analyze every 0.2 seconds (every 8 frames at 40 FPS)
                if (len(frame_buffer) == BUFFER_SIZE and 
                    current_time - last_analysis_time >= ANALYSIS_INTERVAL):
                    
                    prediction_logger.info(f"ANALYSIS_START - Time: {current_time:.2f}s, Frame: {frame_count}, Buffer size: {len(frame_buffer)}")
                    
                    sequence = np.array(frame_buffer)
                    # Apply the same preprocessing as in training
                    if use_velocity and len(sequence) > 2:
                        sequence = add_velocity_features(sequence)
                    if use_statistics and len(sequence) > 1:
                        from pose_utils import add_statistical_features
                        sequence = add_statistical_features(sequence)
                    if use_ratios and len(sequence) > 1:
                        from pose_utils import add_ratio_features
                        sequence = add_ratio_features(sequence, focus_indices)
                    
                    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
                    lengths = torch.tensor([BUFFER_SIZE])
                    with torch.no_grad():
                        output = model(sequence_tensor, lengths)
                        predicted = torch.argmax(output, dim=1).item()
                        confidence = torch.softmax(output, dim=1)[0][predicted].item()
                    predicted_label = LABELS[predicted]
                    
                    # Log prediction details
                    prediction_logger.info(f"PREDICTION - Time: {current_time:.2f}s, Label: {predicted_label}, Confidence: {confidence:.3f}, Frame: {frame_count}")

                    # Add prediction to history
                    prediction_history.append({
                        'label': predicted_label,
                        'confidence': confidence,
                        'timestamp': current_time
                    })

                    # Check for agreement in recent predictions
                    final_prediction = "idle"
                    final_confidence = 0.0
                    rep_completed = True
                    state_message = "no_agreement"

                    if len(prediction_history) >= MIN_AGREEMENT:
                        # Count occurrences of each label in recent predictions
                        label_counts = {}
                        high_confidence_predictions = []
                        
                        for pred in prediction_history:
                            label = pred['label']
                            conf = pred['confidence']
                            label_counts[label] = label_counts.get(label, 0) + 1
                            
                            if conf >= MIN_CONFIDENCE_THRESHOLD:
                                high_confidence_predictions.append(pred)
                        
                        # Find the most common label with high confidence
                        if high_confidence_predictions:
                            best_label = max(high_confidence_predictions, 
                                           key=lambda x: (label_counts[x['label']], x['confidence']))
                            
                            # Check if we have enough agreement
                            if label_counts[best_label['label']] >= MIN_AGREEMENT:
                                final_prediction = best_label['label']
                                final_confidence = best_label['confidence']
                                
                                # Update state machine
                                rep_completed, state_message = state_machine.update(final_prediction, final_confidence, current_time)
                                
                                # Update rep counts based on state machine
                                if rep_completed:
                                    rep_counts["good"] = state_machine.rep_count
                                
                                # Log final decision
                                prediction_logger.info(f"DECISION - Time: {current_time:.2f}s, Final: {final_prediction}, Confidence: {final_confidence:.3f}, Rep completed: {rep_completed}, Message: {state_message}")

                    last_analysis_time = current_time
                    prediction_logger.info(f"ANALYSIS_END - Time: {current_time:.2f}s, Final prediction: {final_prediction}, Confidence: {final_confidence:.3f}")

                    # Clean old predictions (older than 2 seconds)
                    current_predictions = []
                    for pred in prediction_history:
                        if current_time - pred['timestamp'] <= 2.0:
                            current_predictions.append(pred)
                    prediction_history = deque(current_predictions, maxlen=VOTING_WINDOW_SIZE)

                    await websocket.send_json({
                        "type": "prediction",
                        "frame_count": frame_count,
                        "current_time": current_time,
                        "prediction": final_prediction,
                        "confidence": float(final_confidence),
                        "rep_counts": rep_counts,
                        "count_this": rep_completed,
                        "state_machine": {
                            "current_state": state_machine.state,
                            "rep_count": state_machine.rep_count,
                            "state_message": state_message,
                            "status_message": state_machine.get_status_message(current_time),
                            "time_in_state": current_time - state_machine.last_state_change_time,
                            "avg_rep_duration": sum(state_machine.rep_durations) / len(state_machine.rep_durations) if state_machine.rep_durations else 0,
                            "rep_durations": state_machine.rep_durations[-5:] if state_machine.rep_durations else [],  # Last 5 reps
                            "bad_form_counts": state_machine.bad_form_counts
                        },
                        "prediction_history_size": len(prediction_history),
                        "high_confidence_count": len([p for p in prediction_history if p['confidence'] >= MIN_CONFIDENCE_THRESHOLD])
                    })

                await websocket.send_json({
                    "type": "frame_processed",
                    "frame_count": frame_count,
                    "pose_detected": True
                })

            except WebSocketDisconnect:
                print("[DEBUG] WebSocket disconnected")
                break
            except Exception as e:
                print(f"[ERROR] Exception in frame processing: {e}")
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Error processing frame: {str(e)}"
                    })
                except:
                    break

    except Exception as e:
        prediction_logger.error(f"ERROR - WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"WebSocket error: {str(e)}"
            })
        except:
            pass
    finally:
        if holistic is not None:
            holistic.close()
        prediction_logger.info(f"SESSION_END - Prediction session ended for exercise: {exercise_name}")
        print("[DEBUG] MediaPipe holistic closed.")

def extract_keypoints_xyz(landmarks):
    """Extract keypoint coordinates for additional features"""
    indices = [
        mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER,
        mp.solutions.holistic.PoseLandmark.RIGHT_ELBOW,
        mp.solutions.holistic.PoseLandmark.RIGHT_WRIST,
        mp.solutions.holistic.PoseLandmark.RIGHT_HIP,
        mp.solutions.holistic.PoseLandmark.RIGHT_KNEE,
        mp.solutions.holistic.PoseLandmark.RIGHT_ANKLE,
        mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER,
        mp.solutions.holistic.PoseLandmark.LEFT_ELBOW,
        mp.solutions.holistic.PoseLandmark.LEFT_WRIST,
        mp.solutions.holistic.PoseLandmark.LEFT_HIP,
        mp.solutions.holistic.PoseLandmark.LEFT_KNEE,
        mp.solutions.holistic.PoseLandmark.LEFT_ANKLE,
    ]
    coords = []
    for idx in indices:
        pt = landmarks.landmark[idx]
        coords.extend([pt.x, pt.y, pt.z])
    return coords

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
