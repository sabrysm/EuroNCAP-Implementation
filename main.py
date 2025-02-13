import cv2
from decision_layer import DecisionLayer
import winsound

def main():
    # Test audio
    try:
        winsound.PlaySound(None, winsound.SND_PURGE)  # Clear any playing sounds
    except Exception as e:
        print(f"Warning: Audio device not available - {e}")

    # Initialize the decision layer
    decision_layer = DecisionLayer(model_path='./best.pt')

    # Open video capture
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame through decision layer
        processed_frame, detection_results = decision_layer.process_frame(frame)

        # Display the frame
        cv2.imshow('EuroNCAP Implementation', processed_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
