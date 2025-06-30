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

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typing import List
from pose_utils import LSTMClassifier, extract_angles, extract_sequence_from_video, CNN_LSTM_Classifier, LSTM_Transformer_Classifier, extract_keypoints_xyz, get_angle_indices_by_parts

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
        model = load_model(exercise_name)
        
        # Prepare data for model
        sequence_tensor = torch.tensor([sequence], dtype=torch.float32)
        lengths = torch.tensor([len(sequence)])
        
        # Prediction
        with torch.no_grad():
            outputs = model(sequence_tensor, lengths)
            _, predicted = torch.max(outputs, 1)
            
        # Cleanup
        os.remove(video_path)
        
        return {
            'prediction': 'good' if predicted.item() == 0 else 'bad',
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


@app.websocket("/ws/video-analysis-base64/{exercise_name}")
async def websocket_video_analysis_base64(websocket: WebSocket, exercise_name: str):
    await websocket.accept()
    holistic = None

    try:
        # ==== הגדר פה בדיוק כמו באימון! ====
        # תעדכן focus_parts בדיוק כמו ששימשת בשורת האימון
        focus_parts = ['right_knee', 'torso']
        use_keypoints = True  # כי השתמשת בפרמטר --use_keypoints
        focus_indices = get_angle_indices_by_parts(focus_parts)
        input_size = len(focus_indices) + (12 * 3 if use_keypoints else 0)

        print(f"[DEBUG] Loading model for exercise: {exercise_name}")
        # עדכן לפי איך שטענת את המודל שלך
        model = load_model(exercise_name, model_type='lstm', input_size=input_size, num_classes=2, bidirectional=True)
        model.eval()

        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("[DEBUG] Model and MediaPipe loaded. Waiting for frames...")

        await websocket.send_json({
            "type": "status",
            "message": f"Ready to analyze {exercise_name} exercise (base64)",
            "model_loaded": True
        })

        window_size = 20
        step = 3
        frame_buffer = []
        frame_count = 0
        rep_count = 0
        prev_label = 0

        while True:
            try:
                data = await websocket.receive_text()
                if data.startswith('data:image/'):
                    data = data.split(',')[1]
                binary_data = base64.b64decode(data)
                nparr = np.frombuffer(binary_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    print("[ERROR] Frame decode failed (cv2.imdecode returned None)")
                    continue

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)

                if not results.pose_landmarks:
                    await websocket.send_json({
                        "type": "pose_missing",
                        "frame_count": frame_count
                    })
                    continue

                # === בדיוק כמו באימון! ===
                angles = extract_angles(results.pose_landmarks)
                features = [angles[i] for i in focus_indices] if focus_indices else angles
                if use_keypoints:
                    keypoints = extract_keypoints_xyz(results.pose_landmarks)
                    features += keypoints

                # השלמת אפסים, קיצוץ אקסטרה, תמיד בגודל input_size
                if len(features) < input_size:
                    features.extend([0.0] * (input_size - len(features)))
                elif len(features) > input_size:
                    features = features[:input_size]

                # debug קצר לוודא שהכל טוב
                # print("len(features):", len(features))

                frame_buffer.append(features)
                frame_count += 1
                if len(frame_buffer) > window_size:
                    frame_buffer = frame_buffer[-window_size:]

                # נריץ מודל כל STEP פריימים בלבד
                if len(frame_buffer) == window_size and frame_count % step == 0:
                    sequence = np.array(frame_buffer)
                    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
                    lengths = torch.tensor([window_size])
                    with torch.no_grad():
                        output = model(sequence_tensor, lengths)
                        predicted = torch.argmax(output, dim=1).item()
                        confidence = torch.softmax(output, dim=1)[0][predicted].item()
                    prediction = "good" if predicted == 1 else "bad"

                    # === ספירת חזרות: מעבר bad→good ===
                    if predicted == 1 and prev_label == 0:
                        rep_count += 1
                        await websocket.send_json({
                            "type": "rep_detected",
                            "rep_count": rep_count,
                            "quality": prediction,
                            "confidence": float(confidence),
                            "frame": frame_count
                        })
                    prev_label = predicted

                    await websocket.send_json({
                        "type": "prediction",
                        "frame_count": frame_count,
                        "prediction": prediction,
                        "confidence": float(confidence),
                        "rep_count": rep_count
                    })

                # הודעה תמידית לעדכון
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
        print(f"[ERROR] WebSocket error: {e}")
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
        print("[DEBUG] MediaPipe holistic closed.")

async def websocket_video_analysis_base64(websocket: WebSocket, exercise_name: str):
    """WebSocket endpoint for real-time video analysis using base64 encoded frames (with debug prints)"""
    await websocket.accept()
    holistic = None

    try:
        print(f"[DEBUG] Loading model for exercise: {exercise_name}")
        model = load_model(exercise_name, model_type='lstm', input_size=51, num_classes=2, bidirectional=True)
        model.eval()

        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("[DEBUG] Model and MediaPipe loaded. Waiting for frames...")

        await websocket.send_json({
            "type": "status",
            "message": f"Ready to analyze {exercise_name} exercise (base64)",
            "model_loaded": True
        })

        frame_buffer = []
        frame_count = 0

        while True:
            try:
                data = await websocket.receive_text()
                print("[DEBUG] Received frame from client")
                try:
                    if data.startswith('data:image/'):
                        data = data.split(',')[1]
                    binary_data = base64.b64decode(data)
                    nparr = np.frombuffer(binary_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except Exception as e:
                    print(f"[ERROR] Failed to decode base64 frame: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Failed to decode base64 frame: {str(e)}"
                    })
                    continue

                if frame is None:
                    print("[ERROR] Frame decode failed (cv2.imdecode returned None)")
                    continue

                print("[DEBUG] Frame decoded successfully. Processing with MediaPipe...")
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)

                if not results.pose_landmarks:
                    print("[DEBUG] No pose detected in frame.")
                    continue

                print("[DEBUG] Pose detected!")
                angles = extract_angles(results.pose_landmarks)
                if not angles:
                    print("[DEBUG] No angles extracted from pose.")
                    continue
                keypoints = extract_keypoints_xyz(results.pose_landmarks)
                features = angles + keypoints
                if len(features) < 51:
                    features.extend([0.0] * (51 - len(features)))
                elif len(features) > 51:
                    features = features[:51]
                frame_buffer.append(features)
                frame_count += 1
                print(f"[DEBUG] Frame {frame_count} added to buffer. Buffer size: {len(frame_buffer)}")
                if frame_count % 10 == 0 and len(frame_buffer) >= 5:
                    print(f"[DEBUG] Running prediction on buffer of size {len(frame_buffer)}")
                    sequence = np.array(frame_buffer)
                    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
                    lengths = torch.tensor([len(frame_buffer)])
                    with torch.no_grad():
                        output = model(sequence_tensor, lengths)
                        predicted = torch.argmax(output, dim=1).item()
                        confidence = torch.softmax(output, dim=1)[0][predicted].item()
                    prediction = "good" if predicted == 1 else "bad"
                    print(f"[DEBUG] Prediction: {prediction}, Confidence: {confidence}")
                    await websocket.send_json({
                        "type": "prediction",
                        "frame_count": frame_count,
                        "prediction": prediction,
                        "confidence": float(confidence),
                        "buffer_size": len(frame_buffer)
                    })
                    frame_buffer = frame_buffer[-20:]

                await websocket.send_json({
                    "type": "frame_processed",
                    "frame_count": frame_count,
                    "pose_detected": results.pose_landmarks is not None
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
        print(f"[ERROR] WebSocket error: {e}")
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
