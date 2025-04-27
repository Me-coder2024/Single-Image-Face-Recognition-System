import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image


def recognize_and_track_movement(reference_image_path, video_source=0, threshold=0.7):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(keep_all=True, device=device)
    facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)


    reference_image = Image.open(reference_image_path)
    reference_faces = mtcnn(reference_image)

    if reference_faces is None:
        print("No face detected in the reference image. Exiting.")
        return

    # Extract the first detected face and reshape to [1, 3, 160, 160]
    reference_face = reference_faces[0].unsqueeze(0) if len(reference_faces.shape) == 4 else reference_faces.unsqueeze(
        0)
    reference_face = reference_face.to(device)
    reference_embedding = facenet(reference_face).detach().cpu()

    # Open webcam or video source
    video_capture = cv2.VideoCapture(video_source)
    tracker = None
    tracking_box = None
    prev_center = None  # To store the center of the face in the previous frame

    print("Press 'q' to exit.")

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb_frame)

        if tracker is None or tracking_box is None:
            # Detect faces
            faces = mtcnn(pil_frame)
            if faces is not None:
                embeddings = facenet(faces.to(device)).detach().cpu()
                for embedding, box in zip(embeddings, mtcnn.detect(pil_frame)[0]):
                    # Compute cosine similarity
                    similarity = torch.cosine_similarity(reference_embedding, embedding.unsqueeze(0))
                    label = "Not Matched"
                    color = (0, 0, 255)  # Red

                    if similarity > threshold:
                        label = "Matched!"
                        color = (0, 255, 0)  # Green
                        tracking_box = tuple(map(int, box))
                        tracker = cv2.TrackerKCF_create()  # Initialize tracker
                        tracker.init(frame, tracking_box)  # Initialize with detected face
                        break

                    # Draw a rectangle around the face
                    x1, y1, x2, y2 = [int(v) for v in box]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            # Update the tracker
            success, tracking_box = tracker.update(frame)
            if success:
                x, y, w, h = map(int, tracking_box)
                center = (x + w // 2, y + h // 2)  # Calculate the center of the face

                # Draw the tracking box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow rectangle for tracking
                cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                # Calculate movement if previous center exists
                if prev_center is not None:
                    dx = center[0] - prev_center[0]
                    dy = center[1] - prev_center[1]
                    movement_speed = np.sqrt(dx ** 2 + dy ** 2)  # Calculate the magnitude of movement

                    # Display movement info
                    movement_info = f"Speed: {movement_speed:.2f}, Dx: {dx}, Dy: {dy}"
                    cv2.putText(frame, movement_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                prev_center = center  # Update the previous center
            else:
                print("Tracking failed. Resetting tracker.")
                tracker = None
                tracking_box = None
                prev_center = None

        # Display the resulting frame
        cv2.imshow("Face Recognition and Movement Tracking", frame)

        # Exit the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close the window
    video_capture.release()
    cv2.destroyAllWindows()


# Usage
reference_image_path = r"C:\Users\mecod\PycharmProjects\Team Fresher\image.jpg"  # Replace with your reference image path
recognize_and_track_movement(reference_image_path)
