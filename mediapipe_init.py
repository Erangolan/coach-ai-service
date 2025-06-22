import os
import sys
import logging
import warnings

def init_mediapipe():
    """Initialize MediaPipe with proper log suppression"""
    # Set critical environment variables before importing mediapipe
    os.environ['GLOG_minloglevel'] = '3'
    os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['GLOG_logtostderr'] = '0'
    os.environ['GLOG_log_dir'] = '/dev/null'
    os.environ['GLOG_stderrthreshold'] = '3'
    os.environ['MEDIAPIPE_DISABLE_PERCEPTION'] = '1'
    
    # Suppress all warnings
    warnings.filterwarnings('ignore')
    
    # Configure logging before importing mediapipe
    logging.getLogger('mediapipe').setLevel(logging.CRITICAL)
    logging.getLogger('mediapipe').propagate = False
    logging.getLogger('mediapipe').disabled = True
    
    # Now import mediapipe
    import mediapipe as mp
    
    # Configure MediaPipe solutions
    mp.solutions.drawing_utils.DrawingSpec = lambda *args, **kwargs: None
    
    return mp

# Initialize MediaPipe when this module is imported
mp = init_mediapipe() 