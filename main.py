from fastapi import FastAPI, UploadFile, File, Form, Depends
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

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typing import List
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from pose_utils import LSTMClassifier, extract_angles, extract_sequence_from_video, CNN_LSTM_Classifier, LSTM_Transformer_Classifier

DATABASE_URL = "postgresql://erangolan:eran1234@localhost:5432/exercise_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

app = FastAPI()

# MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Important: Same label mapping as in training
LABELS = ["squat_good", "squat_bad"]
label_to_idx = {label: i for i, label in enumerate(LABELS)}

def load_model(exercise_name, model_type='lstm', input_size=40, num_classes=2, bidirectional=True):
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
def classify_video(file: UploadFile = File(...)):
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
    model = load_model(exercise_name)
    input_tensor = sequence.unsqueeze(0)  # [1, seq_len, 9]
    lengths = torch.tensor([sequence.shape[0]])
    output = model(input_tensor, lengths)
    predicted = torch.argmax(output, dim=1).item()
    predicted_label = LABELS[predicted]

    return {"prediction": predicted_label}

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
        
    video_file = request.files['video']
    exercise_name = request.form.get('exercise', 'right-knee-to-90-degrees')
    
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not video_file.filename.endswith('.mp4'):
        return jsonify({'error': 'Only MP4 files are supported'}), 400
    
    # שמירת הקובץ
    filename = secure_filename(video_file.filename)
    video_path = os.path.join('uploads', filename)
    os.makedirs('uploads', exist_ok=True)
    video_file.save(video_path)
    
    try:
        # חילוץ רצף הזוויות
        sequence = extract_sequence_from_video(video_path)
        if not sequence:
            return jsonify({'error': 'No valid poses detected in video'}), 400
        
        # טעינת המודל
        model = load_model(exercise_name)
        
        # הכנת הנתונים למודל
        sequence_tensor = torch.tensor([sequence], dtype=torch.float32)
        lengths = torch.tensor([len(sequence)])
        
        # חיזוי
        with torch.no_grad():
            outputs = model(sequence_tensor, lengths)
            _, predicted = torch.max(outputs, 1)
            
        # ניקוי
        os.remove(video_path)
        
        return jsonify({
            'prediction': 'good' if predicted.item() == 0 else 'bad',
            'confidence': torch.softmax(outputs, dim=1)[0][predicted].item()
        })
        
    except Exception as e:
        if os.path.exists(video_path):
            os.remove(video_path)
        return jsonify({'error': str(e)}), 500

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
        # Extract sequence from video using the correct pipeline
        # The model was trained with input_size=51, so we need to match that
        sequence = extract_sequence_from_video(video_path, use_keypoints=True)
        if sequence is None or len(sequence) == 0:
            raise HTTPException(status_code=400, detail="No valid pose detected in video.")
        
        # Ensure the sequence has the correct number of features
        if sequence.shape[1] != 51:
            # Pad or truncate to match the expected input size
            if sequence.shape[1] < 51:
                # Pad with zeros
                padding = np.zeros((sequence.shape[0], 51 - sequence.shape[1]))
                sequence = np.hstack([sequence, padding])
            else:
                # Truncate to 51 features
                sequence = sequence[:, :51]
        
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)

        # Load model for the given exercise
        try:
            # Your last training was with LSTM model, but it used input_size=51
            model = load_model(exercise_name, model_type='lstm', input_size=51, num_classes=2, bidirectional=True)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Model for exercise '{exercise_name}' not found: {str(e)}")

        # Prepare input for model
        input_tensor = sequence_tensor.unsqueeze(0)  # [1, seq_len, features]
        lengths = torch.tensor([sequence_tensor.shape[0]])
        with torch.no_grad():
            output = model(input_tensor, lengths)
            predicted = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1)[0][predicted].item()
        predicted_label = "good" if predicted == 1 else "bad"
        return {"prediction": predicted_label, "confidence": confidence}
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

if __name__ == '__main__':
    app.run(debug=True)
