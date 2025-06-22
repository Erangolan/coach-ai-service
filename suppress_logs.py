import os
import logging
import warnings
import sys
import io

def suppress_logs():
    """Suppress verbose logs from various libraries"""
    
    # Set environment variables before anything else
    os.environ['GLOG_minloglevel'] = '3'  # Only show fatal errors
    os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['AUTOGRAPH_VERBOSITY'] = '0'
    os.environ['GLOG_logtostderr'] = '0'
    os.environ['GLOG_log_dir'] = '/dev/null'
    os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
    os.environ['PYTHONWARNINGS'] = 'ignore'
    
    # Redirect both stderr and stdout to devnull
    class DummyStream:
        def write(self, x): pass
        def flush(self): pass
        def fileno(self): return -1
    
    # Keep a reference to the original streams
    original_stderr = sys.stderr
    original_stdout = sys.stdout
    
    # Replace streams with dummy ones
    sys.stderr = DummyStream()
    sys.stdout = DummyStream()
    
    # Suppress all Python warnings
    warnings.filterwarnings('ignore')
    
    # Configure root logger to most restrictive level
    logging.getLogger().setLevel(logging.CRITICAL)
    
    # Configure specific loggers
    loggers = [
        'mediapipe',
        'tensorflow',
        'torch',
        'absl',
        'PIL',
        'matplotlib',
        'numpy',
        'sklearn',
        'pandas'
    ]
    
    for name in loggers:
        logger = logging.getLogger(name)
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False
        logger.disabled = True
        
    # Disable all handlers on the root logger
    logging.getLogger().handlers = []
    
    # Create a null handler and add it to the root logger
    null_handler = logging.NullHandler()
    logging.getLogger().addHandler(null_handler)
    
    def restore_streams():
        """Restore original stdout and stderr streams"""
        sys.stderr = original_stderr
        sys.stdout = original_stdout
    
    return restore_streams

if __name__ == "__main__":
    restore_fn = suppress_logs() 