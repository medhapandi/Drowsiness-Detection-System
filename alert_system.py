import pygame
import time
import cv2
import os

class AlertSystem:
    def __init__(self):
        # Initialize pygame for sound alerts
        try:
            pygame.mixer.init()
        except Exception:
            # If audio initialization fails, keep going without sound
            print("Warning: pygame mixer init failed — continuing without audio")

        # Load alert sounds (optional)
        self.alert_sound = None
        self.is_playing = False
        try:
            if os.path.exists("alarm.wav"):
                self.alert_sound = pygame.mixer.Sound("alarm.wav")
            else:
                print("Warning: alarm.wav not found — sound disabled")
        except Exception as e:
            print("Warning: could not load alarm sound:", e)

    def trigger_alert(self, alert_type, frame):
        """Trigger appropriate alert based on type"""
        print(f"ALERT: {alert_type} detected!")

        # Visual alert on frame
        cv2.putText(frame, "DROWSINESS DETECTED!", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Sound alert (if available)
        if self.alert_sound is not None and not self.is_playing:
            try:
                self.is_playing = True
                self.alert_sound.play()
                # Stop sound after 2 seconds
                pygame.time.delay(2000)
                self.alert_sound.stop()
            except Exception as e:
                print("Warning: failed playing alert sound:", e)
            finally:
                self.is_playing = False

    def cleanup(self):
        """Cleanup resources"""
        try:
            pygame.mixer.quit()
        except Exception:
            pass