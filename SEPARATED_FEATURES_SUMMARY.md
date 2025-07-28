# LSTM + GNN עם הפרדת תכונות מרחביות וטמפורליות

## 🎯 **השינוי שביצענו**

שינינו את הארכיטקטורה כך שה-**GNN** מקבל רק **תכונות מרחביות** וה-**LSTM** מקבל את **פלט ה-GNN + תכונות טמפורליות**.

## 📊 **הארכיטקטורה החדשה**

```
Input Video
    ↓
[Feature Extraction]
    ↓
┌─────────────────┬─────────────────┐
│ SPATIAL FEATURES│ TEMPORAL FEATURES│
│ (for GNN)       │ (for LSTM)      │
│                 │                 │
│ • Keypoints     │ • Velocity      │
│ • Angles        │ • Acceleration  │
│ • Distances     │ • Std Dev       │
│                 │ • Angle Changes │
│                 │ • Ratios        │
└─────────────────┴─────────────────┘
    ↓                    ↓
[GNN Processing]    [Temporal Features]
    ↓                    ↓
[Spatial Understanding]  │
    ↓                    │
    └─────────┬─────────┘
              ↓
    [LSTM Processing]
              ↓
    [Final Classification]
```

## 🔧 **פירוט התכונות**

### **תכונות מרחביות (GNN) - 67 תכונות**

#### **1. Keypoints (36 תכונות)**
```python
# 12 מפרקים × 3 קואורדינטות (x, y, z)
keypoints = [
    right_shoulder_x, right_shoulder_y, right_shoulder_z,
    right_elbow_x, right_elbow_y, right_elbow_z,
    right_wrist_x, right_wrist_y, right_wrist_z,
    # ... וכן הלאה לכל 12 המפרקים
]
```

#### **2. Angles (15 תכונות)**
```python
angles = [
    right_arm_angle,      # כתף-מרפק-כף יד
    left_arm_angle,       # כתף-מרפק-כף יד
    right_leg_angle,      # ירך-ברך-קרסול
    left_leg_angle,       # ירך-ברך-קרסול
    # ... וכן הלאה
]
```

#### **3. Distances (16 תכונות)**
```python
distances = [
    shoulder_to_elbow_distance,
    elbow_to_wrist_distance,
    hip_to_knee_distance,
    knee_to_ankle_distance,
    # ... וכן הלאה
]
```

### **תכונות טמפורליות (LSTM) - ~200 תכונות**

#### **1. Velocity (67 תכונות)**
```python
velocity = current_position - previous_position
# לכל התכונות המרחביות
```

#### **2. Acceleration (67 תכונות)**
```python
acceleration = current_velocity - previous_velocity
# שינוי המהירות
```

#### **3. Standard Deviation (67 תכונות)**
```python
std_features = np.std(sequence, axis=0)
# סטיית תקן של כל התכונות לאורך הזמן
```

#### **4. Angle Changes (15 תכונות)**
```python
angle_changes = np.diff(angles, axis=0)
# קצב השינוי בזוויות
```

#### **5. Ratios (2 תכונות)**
```python
ratios = [
    knee_angle / torso_angle,
    shoulder_angle / hip_angle
]
```

## 🏗️ **הארכיטקטורה החדשה**

### **1. GNN Layer**
```python
# קלט: (batch_size, seq_len, 67) → (batch_size * seq_len, 12, 5.58)
# 67 תכונות מרחביות מחולקות ל-12 מפרקים
spatial_features = spatial_features.reshape(batch_size * seq_len, 12, 5.58)

# GNN מעבד כל מפרק בנפרד
gnn_output = gnn_layers(spatial_features, adjacency_matrix)
# פלט: (batch_size, seq_len, 12 * 64) = (batch_size, seq_len, 768)
```

### **2. LSTM Layer**
```python
# שילוב פלט GNN + תכונות טמפורליות
combined_features = torch.cat([gnn_output, temporal_features], dim=2)
# (batch_size, seq_len, 768 + 200) = (batch_size, seq_len, 968)

# LSTM מעבד את הרצף
lstm_output = lstm(combined_features)
# פלט: (batch_size, 256) - hidden state סופי
```

### **3. Classification Layer**
```python
# סיווג סופי
output = fc_layers(lstm_output)
# פלט: (batch_size, 5) - 5 קטגוריות
```

## 🎯 **היתרונות של השינוי**

### **1. הפרדת אחריות ברורה**
- **GNN**: מתמקד רק בקשרים מרחביים בין מפרקים
- **LSTM**: מתמקד רק בדינמיקה טמפורלית של התנועה

### **2. יעילות חישובית**
- GNN מעבד פחות תכונות (67 במקום 200+)
- LSTM מקבל מידע מעובד ומאורגן

### **3. יכולת פרשנות**
- קל יותר להבין מה ה-GNN לומד
- קל יותר להבין מה ה-LSTM לומד

### **4. גמישות**
- אפשר לשנות תכונות מרחביות בנפרד
- אפשר לשנות תכונות טמפורליות בנפרד

## 📝 **דוגמה לשימוש**

### **אימון המודל**
```python
model = LSTM_GNN_Classifier(
    spatial_input_size=67,    # תכונות מרחביות
    temporal_input_size=200,  # תכונות טמפורליות
    hidden_size=128,
    num_classes=5,
    num_joints=12,
    gnn_hidden=64,
    num_gnn_layers=2
)

# Forward pass
output = model(spatial_features, temporal_features, lengths)
```

### **מיצוי תכונות**
```python
# מיצוי תכונות מרחביות
spatial_features = extract_spatial_features(landmarks)
# keypoints + angles + distances

# מיצוי תכונות טמפורליות
temporal_features = extract_temporal_features(sequence)
# velocity + acceleration + std + angle_changes + ratios
```

## 🔍 **השוואה למודל הקודם**

| מאפיין | מודל קודם | מודל חדש |
|---------|------------|-----------|
| **GNN Input** | כל התכונות | רק תכונות מרחביות |
| **LSTM Input** | פלט GNN | פלט GNN + תכונות טמפורליות |
| **הפרדה** | מעורבב | ברורה |
| **יעילות** | נמוכה | גבוהה |
| **פרשנות** | קשה | קלה |

## 🚀 **התוצאה הסופית**

עכשיו יש לנו מודל שמשלב:
- ✅ **הבנה מרחבית מתקדמת** (GNN על תכונות מרחביות)
- ✅ **הבנה טמפורלית מתקדמת** (LSTM על תכונות טמפורליות)
- ✅ **הפרדת אחריות ברורה**
- ✅ **יעילות חישובית משופרת**
- ✅ **יכולת פרשנות טובה יותר**

זה מודל הרבה יותר חכם ומדויק לזיהוי תרגילים! 