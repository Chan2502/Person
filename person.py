import cv2
import numpy as np
from ultralytics import YOLO
from image_dehazer import image_dehazer
model = YOLO('yolov8n.pt')
dehazer = image_dehazer(
    airlightEstimation_windowSze=15,
    boundaryConstraint_windowSze=3,
    C0=20,
    C1=300,
    regularize_lambda=0.1,
    sigma=0.5,
    delta=0.85,
    showHazeTransmissionMap=Falseobs
)
def dehaze_and_detect():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    frame_count = 0

    print("Starting real-time dehazing and person counting. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        frame_resized = cv2.resize(frame, (500, 400))
        if frame_count % int(fps) == 0:
            dehazed_frame, _ = dehazer.remove_haze(frame_resized)
            results = model(dehazed_frame)[0]

            person_count = 0
           
            for box in results.boxes:
                label = int(box.cls[0])
                if label == 0:  
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cv2.rectangle(dehazed_frame, (x1, y1),
                                  (x2, y2), (0, 255, 0), 2)
                    person_count += 1
            overlay_text = f"Detected persons: {person_count}"
            cv2.putText(dehazed_frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Dehazed Person Detection', dehazed_frame)
            print(overlay_text)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    dehaze_and_detect()
