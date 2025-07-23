# Coach AI Service - Exercise Recognition System

A real-time exercise recognition system using computer vision and deep learning to analyze workout form and count repetitions.

## 🚀 Features

- **Real-time Exercise Recognition**: Analyze exercises in real-time using webcam
- **Multiple Model Architectures**: CNN-LSTM, LSTM, and LSTM-Transformer models
- **State Machine Rep Counting**: Intelligent repetition counting with form validation
- **Comprehensive Logging**: Detailed logs for debugging and analysis
- **WebSocket Support**: Real-time communication for live analysis
- **Database Integration**: PostgreSQL for storing exercise data
- **Form Validation**: Detects bad form states (bad angles, wrong positions)

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- PostgreSQL database
- Webcam for real-time analysis
- CUDA-compatible GPU (optional, for faster training)

### Dependencies
```bash
pip install -r requirements.txt
```

## 🗄️ Database Setup

1. **Install PostgreSQL** and create a database:
```sql
CREATE DATABASE exercise_db;
CREATE USER erangolan WITH PASSWORD 'eran1234';
GRANT ALL PRIVILEGES ON DATABASE exercise_db TO erangolan;
```

2. **Update database connection** in `main.py`:
```python
DATABASE_URL = "postgresql://username:password@localhost:5432/exercise_db"
```

## 🏋️ Training Your Own Model

### 1. Prepare Your Dataset

Organize your exercise videos in the following structure:
```
data/
├── right-knee-to-90-degrees/
│   ├── good/
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   │   └── ...
│   ├── bad-left-angle/
│   │   ├── video1.mp4
│   │   └── ...
│   ├── bad-right-angle/
│   │   ├── video1.mp4
│   │   └── ...
│   ├── bad-lower-knee/
│   │   ├── video1.mp4
│   │   └── ...
│   └── idle/
│       ├── video1.mp4
│       └── ...
```

### 2. Configure Training Parameters

Edit `train_lstm.py` to customize your training:

```python
# Model configuration
MODEL_TYPE = 'cnn_lstm'  # Options: 'cnn_lstm', 'lstm', 'lstm_transformer'
BIDIRECTIONAL = True
HIDDEN_SIZE = 128
NUM_LAYERS = 2

# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
TRAIN_SPLIT = 0.8

# Feature extraction
USE_KEYPOINTS = True
USE_VELOCITY = True
USE_STATISTICS = True
USE_RATIOS = True
FOCUS_PARTS = ['right_knee', 'right_shoulder', 'torso']
```

### 3. Run Training

```bash
# Train a new model
python train_lstm.py --exercise_name "right-knee-to-90-degrees" --model_type "cnn_lstm"

# Continue training from checkpoint
python train_lstm.py --exercise_name "right-knee-to-90-degrees" --model_type "cnn_lstm" --resume
```

### 4. Training Output

The training will create:
- `models/right-knee-to-90-degrees_cnn_lstm.pth` - Trained model
- `training_logs/` - Training metrics and plots
- `confusion_matrix.png` - Model performance visualization

## 🎯 Real-Time Analysis

### 1. Start the Server

```bash
# Start the FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. WebSocket Connection

Connect to the real-time analysis endpoint:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/video-analysis-base64/right-knee-to-90-degrees');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Analysis result:', data);
};
```

### 3. State Machine Parameters

The real-time system uses these parameters:
```python
ANALYSIS_INTERVAL = 0.2  # Analysis frequency (seconds)
VOTING_WINDOW_SIZE = 5   # Number of predictions to consider
MIN_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for predictions
MIN_TIME_BETWEEN_REPS = 0.1  # Minimum time between rep counting
MIN_AGREEMENT = 3  # Minimum agreement for final decision

# State machine parameters
MIN_STATE_DURATION = 0.1  # Minimum time in each state
MIN_REP_DURATION = 0.2    # Minimum rep duration
MAX_REP_DURATION = 2.0    # Maximum rep duration
STATE_TIMEOUT = 1.0       # Maximum time in any state
```

## 📊 API Endpoints

### REST Endpoints

#### Upload Reference Exercise
```bash
curl -X POST "http://localhost:8000/upload_reference/" \
  -F "exercise_name=right-knee-to-90-degrees" \
  -F "file=@reference_video.mp4"
```

#### Evaluate Exercise
```bash
curl -X POST "http://localhost:8000/evaluate_exercise/" \
  -F "exercise_name=right-knee-to-90-degrees" \
  -F "file=@test_video.mp4"
```

#### Classify Video
```bash
curl -X POST "http://localhost:8000/classify/" \
  -F "exercise_name=right-knee-to-90-degrees" \
  -F "file=@test_video.mp4"
```

#### Analyze Video
```bash
curl -X POST "http://localhost:8000/analyze/" \
  -F "exercise_name=right-knee-to-90-degrees" \
  -F "file=@test_video.mp4"
```

### WebSocket Endpoints

#### Real-time Video Analysis
```javascript
// Connect to real-time analysis
const ws = new WebSocket('ws://localhost:8000/ws/video-analysis-base64/right-knee-to-90-degrees');

// Send base64 encoded video frames
ws.send(JSON.stringify({
    type: "video_frame",
    data: base64EncodedFrame
}));
```

## 🔍 Understanding the State Machine

### States
- **`idle`**: Ready for next repetition
- **`good`**: Correct form detected
- **`bad-left-angle`**: Left angle is incorrect
- **`bad-right-angle`**: Right angle is incorrect
- **`bad-lower-knee`**: Knee position is too low

### Rep Counting Logic
1. **Start**: `idle` → `good` (marks rep start)
2. **Complete**: Any transition out of `good` counts as rep completion
3. **Timeout**: After 1 second in any state, return to `idle`

### Example Flow
```
idle → good → bad-left-angle → idle
     ↑     ↑                ↑
   Start  Rep            Complete
         Count
```

## 📝 Logging System

### Log Files
Logs are saved in `logs/predictions_{exercise_name}_{timestamp}.log`

### Log Types
- **PREDICTION**: Raw model predictions
- **STATE**: State machine transitions
- **DECISION**: Final decisions after voting
- **ANALYSIS**: Window analysis start/end
- **ERROR**: System errors

### Example Log
```
2024-01-15 10:30:16 - prediction_right-knee-to-90-degrees - INFO - PREDICTION - Time: 1.20s, Label: good, Confidence: 0.856, Frame: 48
2024-01-15 10:30:16 - prediction_right-knee-to-90-degrees - INFO - STATE - Time: 1.20s, idle → good (rep_started) - Prediction: good (conf: 0.856)
2024-01-15 10:30:16 - prediction_right-knee-to-90-degrees - INFO - DECISION - Time: 1.20s, Final: good, Confidence: 0.856, Rep completed: False
```

## 🛠️ Customization

### Adding New Exercises
1. Create dataset folder structure
2. Add exercise videos for each class
3. Update `LABELS` in `main.py` if needed
4. Train new model with `train_lstm.py`

### Modifying Feature Extraction
Edit `pose_utils.py` to customize:
- Angle calculations
- Keypoint features
- Velocity features
- Statistical features

### Adjusting State Machine
Modify `RepStateMachine` class in `main.py`:
- Timeout durations
- Confidence thresholds
- State transition logic

## 🐛 Troubleshooting

### Common Issues

#### Model Not Loading
```bash
# Check model file exists
ls models/right-knee-to-90-degrees_cnn_lstm.pth

# Verify model architecture matches
python -c "import torch; model = torch.load('models/right-knee-to-90-degrees_cnn_lstm.pth'); print(model.keys())"
```

#### Database Connection Error
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Test connection
psql -h localhost -U erangolan -d exercise_db
```

#### WebSocket Connection Issues
```javascript
// Check server is running
fetch('http://localhost:8000/docs')

// Test WebSocket connection
const ws = new WebSocket('ws://localhost:8000/ws/video-analysis-base64/test');
ws.onopen = () => console.log('Connected!');
ws.onerror = (e) => console.error('Error:', e);
```

### Performance Optimization

#### GPU Acceleration
```python
# Enable CUDA in train_lstm.py
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

#### Batch Processing
```python
# Increase batch size for faster training
BATCH_SIZE = 32  # or higher if memory allows
```

## 📈 Model Performance

### Metrics to Monitor
- **Accuracy**: Overall classification accuracy
- **Precision**: Correct predictions / Total predictions
- **Recall**: Correct predictions / Total actual instances
- **F1-Score**: Harmonic mean of precision and recall

### Improving Performance
1. **More Data**: Add more training videos
2. **Data Augmentation**: Use different angles, lighting
3. **Feature Engineering**: Add more relevant features
4. **Model Architecture**: Try different model types
5. **Hyperparameter Tuning**: Optimize learning rate, batch size

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MediaPipe for pose estimation
- PyTorch for deep learning framework
- FastAPI for web framework
- PostgreSQL for database

---

For more detailed documentation, check the inline comments in the code files. 