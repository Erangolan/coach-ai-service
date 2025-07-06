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
LABELS_MAP = {
    "good": 0,
    "bad-knee-angle": 1,
    "bad-lower-knee": 2,
    "idle": 3,
}
LABELS = list(LABELS_MAP.keys())
label_to_idx = LABELS_MAP

def load_model(exercise_name, model_type='lstm', input_size=40, num_classes=4, bidirectional=True):
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
        focus_parts = ['right_knee', 'torso']
        use_keypoints = True
        focus_indices = get_angle_indices_by_parts(focus_parts)
        input_size = len(focus_indices) + (12 * 3 if use_keypoints else 0)
        
        # Extract sequence from video using the correct pipeline
        sequence = extract_sequence_from_video(video_path, focus_indices=focus_indices, use_keypoints=use_keypoints)
        if sequence is None or len(sequence) == 0:
            raise HTTPException(status_code=400, detail="No valid pose detected in video.")
        
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)

        # Load model for the given exercise
        try:
            model = load_model(exercise_name, model_type='cnn_lstm', input_size=input_size, num_classes=4, bidirectional=True)
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

    try:
        focus_parts = ['right_knee', 'torso']
        use_keypoints = True
        focus_indices = get_angle_indices_by_parts(focus_parts)
        input_size = len(focus_indices) + (12 * 3 if use_keypoints else 0)

        model = load_model(
            exercise_name, model_type='cnn_lstm',
            input_size=input_size, num_classes=4, bidirectional=True
        )
        model.eval()

        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        LABELS = ["good", "bad-knee-angle", "bad-lower-knee", "idle"]
        BUFFER_SIZE = 10    # או 50 אם אתה מעדיף

        frame_buffer = []
        frame_count = 0
        rep_counts = {label: 0 for label in LABELS}

        await websocket.send_json({
            "type": "status",
            "message": f"Ready to analyze {exercise_name} exercise (base64)",
            "model_loaded": True
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

                if len(frame_buffer) == BUFFER_SIZE:
                    sequence = np.array(frame_buffer)
                    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
                    lengths = torch.tensor([BUFFER_SIZE])
                    with torch.no_grad():
                        output = model(sequence_tensor, lengths)
                        predicted = torch.argmax(output, dim=1).item()
                        confidence = torch.softmax(output, dim=1)[0][predicted].item()
                    predicted_label = LABELS[predicted]
                    rep_counts[predicted_label] += 1

                    await websocket.send_json({
                        "type": "prediction",
                        "frame_count": frame_count,
                        "prediction": predicted_label,
                        "confidence": float(confidence),
                        "rep_counts": rep_counts
                    })
                    frame_buffer = []  # מתחילים מחדש

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


@app.websocket("/ws/video-analysis-base64/{exercise_name}")
async def websocket_video_analysis_base64(websocket: WebSocket, exercise_name: str):
    await websocket.accept()
    holistic = None

    try:
        focus_parts = ['right_knee', 'torso']
        use_keypoints = True
        focus_indices = get_angle_indices_by_parts(focus_parts)
        input_size = len(focus_indices) + (12 * 3 if use_keypoints else 0)

        model = load_model(
            exercise_name, model_type='cnn_lstm',
            input_size=input_size, num_classes=4, bidirectional=True
        )
        model.eval()

        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        LABELS = ["good", "bad-knee-angle", "bad-lower-knee", "idle"]
        BUFFER_SIZE = 40
        VOTING_WINDOW = 8  # הצבעות – אפשר גם 1 (בלי majority)

        frame_buffer = deque(maxlen=BUFFER_SIZE)
        pred_window = deque(maxlen=VOTING_WINDOW)
        rep_counts = {label: 0 for label in LABELS}
        frame_count = 0
        last_predicted_label = "idle"
        last_rep_frame = 0

        await websocket.send_json({
            "type": "status",
            "message": f"Ready to analyze {exercise_name} exercise (base64)",
            "model_loaded": True
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

                if len(frame_buffer) == BUFFER_SIZE:
                    sequence = np.array(frame_buffer)
                    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
                    lengths = torch.tensor([BUFFER_SIZE])
                    with torch.no_grad():
                        output = model(sequence_tensor, lengths)
                        predicted = torch.argmax(output, dim=1).item()
                        confidence = torch.softmax(output, dim=1)[0][predicted].item()
                    predicted_label = LABELS[predicted]

                    pred_window.append(predicted_label)
                    # majority voting – או פשוט קח predicted_label אם אתה לא רוצה להחליק
                    voted_label = max(set(pred_window), key=pred_window.count)

                    count_this = False
                    # נחשב חזרה רק כשיש מעבר מ-idle/קטגוריה אחרת
                    if (voted_label != last_predicted_label 
                        and voted_label != "idle" 
                        and frame_count - last_rep_frame > BUFFER_SIZE // 2):
                        rep_counts[voted_label] += 1
                        count_this = True
                        last_predicted_label = voted_label
                        last_rep_frame = frame_count

                    await websocket.send_json({
                        "type": "prediction",
                        "frame_count": frame_count,
                        "prediction": voted_label,
                        "confidence": float(confidence),
                        "rep_counts": rep_counts,
                        "count_this": count_this
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

    import collections
    await websocket.accept()
    holistic = None

    try:
        focus_parts = ['right_knee', 'torso']
        use_keypoints = True
        focus_indices = get_angle_indices_by_parts(focus_parts)
        input_size = len(focus_indices) + (12 * 3 if use_keypoints else 0)

        model = load_model(
            exercise_name, model_type='cnn_lstm',
            input_size=input_size, num_classes=4, bidirectional=True
        )
        model.eval()

        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        LABELS = ["good", "bad-knee-angle", "bad-lower-knee", "idle"]
        BUFFER_SIZE = 10      # החלון המחליק - חזרה אחת
        VOTE_WINDOW = 7       # לכמה תחזיות אחרונות מסתכלים
        frame_buffer = collections.deque(maxlen=BUFFER_SIZE)
        pred_window = collections.deque(maxlen=VOTE_WINDOW)
        last_predicted_label = None
        last_rep_frame = -BUFFER_SIZE
        rep_counts = {label: 0 for label in LABELS}
        frame_count = 0

        await websocket.send_json({
            "type": "status",
            "message": f"Ready to analyze {exercise_name} exercise (base64)",
            "model_loaded": True
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

                if len(frame_buffer) == BUFFER_SIZE:
                    sequence = np.array(frame_buffer)
                    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
                    lengths = torch.tensor([BUFFER_SIZE])
                    with torch.no_grad():
                        output = model(sequence_tensor, lengths)
                        predicted = torch.argmax(output, dim=1).item()
                        confidence = torch.softmax(output, dim=1)[0][predicted].item()
                    predicted_label = LABELS[predicted]

                    # majority voting אם רוצים, אחרת תוכל להוריד
                    pred_window.append(predicted_label)
                    voted_label = collections.Counter(pred_window).most_common(1)[0][0]

                    # תספור חזרה רק אם יש מעבר/לא idle, אבל תשלח prediction תמיד!
                    count_this = (voted_label != last_predicted_label and voted_label != "idle" and frame_count - last_rep_frame > BUFFER_SIZE // 2)
                    if count_this:
                        rep_counts[voted_label] += 1
                        last_predicted_label = voted_label
                        last_rep_frame = frame_count

                    # שולח תמיד את הניבוי (גם אם לא השתנה)
                    await websocket.send_json({
                        "type": "prediction",
                        "frame_count": frame_count,
                        "prediction": voted_label,
                        "confidence": float(confidence),
                        "rep_counts": rep_counts,
                        "count_this": count_this
                    })
                    # הזזה של החלון – הזזת הפריים הראשון בלבד!
                    frame_buffer.popleft()

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
