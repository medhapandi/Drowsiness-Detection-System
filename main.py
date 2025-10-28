import cv2
import time
from drowsiness_detector import DrowsinessDetector
from utils.alert_system import AlertSystem

def main():
    # Initialize components
    detector = DrowsinessDetector()
    alert_system = AlertSystem()
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Starting Drowsiness Detection System...")
    print("Press 'q' to quit")
    
    # Variables for drowsiness tracking
    closed_eyes_start = None
    drowsy_threshold = 2.0  # seconds
    yawn_count = 0
    yawn_start = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect drowsiness
        results = detector.detect_drowsiness(frame)
        
        # Process results
        if results['eyes_closed']:
            if closed_eyes_start is None:
                closed_eyes_start = time.time()
            else:
                eyes_closed_duration = time.time() - closed_eyes_start
                
                # Check if eyes have been closed for too long
                if eyes_closed_duration >= drowsy_threshold:
                    alert_system.trigger_alert("EYES_CLOSED", frame)
                    cv2.putText(frame, "ALERT: EYES CLOSED TOO LONG!", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                               (0, 0, 255), 2)
        else:
            closed_eyes_start = None
            
        # Check for yawning
        if results['yawning']:
            if yawn_start is None:
                yawn_start = time.time()
            else:
                if time.time() - yawn_start > 3:  # Yawn duration threshold
                    yawn_count += 1
                    yawn_start = None
                    
            if yawn_count >= 2:  # Multiple yawns threshold
                alert_system.trigger_alert("YAWNING", frame)
                cv2.putText(frame, "ALERT: EXCESSIVE YAWNING!", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           (0, 0, 255), 2)
        else:
            yawn_start = None
            
        # Display status information
        status_text = f"Eye Aspect Ratio: {results['eye_aspect_ratio']:.2f}"
        cv2.putText(frame, status_text, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if results['yawning']:
            cv2.putText(frame, "Yawning Detected", (10, frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Display frame
        cv2.imshow('Drowsiness Detection System', frame)
        
        # Break on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    alert_system.cleanup()

if __name__ == "__main__":
    main()