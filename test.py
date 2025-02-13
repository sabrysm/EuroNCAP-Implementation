import winsound
import time

# Path to the alert sound
alert_sound_path = r'E:\\College\\Term 10\\Graduation Project 2\\EuroNCAP Implementation\\alert.wav'

def play_alert_sound():
    try:
        winsound.PlaySound(alert_sound_path, winsound.SND_FILENAME)
        print("Alert sound played successfully")
    except Exception as e:
        print(f"Error playing sound: {e}")

# Allowing time for the sound to play
def wait_for_sound_to_finish(duration):
    time.sleep(duration)

# Calling the function to play alert sound and wait for it to finish
play_alert_sound()
