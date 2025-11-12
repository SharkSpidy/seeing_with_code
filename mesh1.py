import cv2
import json
import csv
import time
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# drawing styles (optional)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def normalized_to_pixel_coords(norm_x, norm_y, image_width, image_height):
    # clamp and convert
    x_px = min(max(int(norm_x * image_width), 0), image_width - 1)
    y_px = min(max(int(norm_y * image_height), 0), image_height - 1)
    return x_px, y_px

def run_webcam(max_num_faces=1):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    # FaceMesh: set refine_landmarks=True to get iris landmarks
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=max_num_faces,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        prev_time = time.time()
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                h, w, _ = frame.shape

                # iterate faces
                for fi, face_landmarks in enumerate(results.multi_face_landmarks):
                    # draw the mesh
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=0)
                    )
                    # draw contours and irises for clarity
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec
                    )
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec
                    )

                    # collect landmarks into a list
                    lm_list = []
                    for idx, lm in enumerate(face_landmarks.landmark):
                        x_norm, y_norm, z_norm = lm.x, lm.y, lm.z
                        x_px, y_px = normalized_to_pixel_coords(x_norm, y_norm, w, h)
                        lm_list.append({
                            "index": idx,
                            "x_norm": float(x_norm),
                            "y_norm": float(y_norm),
                            "z_norm": float(z_norm),
                            "x_px": int(x_px),
                            "y_px": int(y_px)
                        })

                    # Example: compute bounding rectangle from landmarks
                    xs = [p["x_px"] for p in lm_list]
                    ys = [p["y_px"] for p in lm_list]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)

                    # Print a few landmarks to console (face 0 only)
                    if fi == 0:
                        # prints mouth tip, left eye, right eye indices as examples
                        # (you can pick any index from MediaPipe Face Mesh landmark map)
                        sample_indices = [1, 33, 263]  # common useful points
                        samples = {si: lm_list[si] for si in sample_indices}
                        print(f"Face {fi} sample landmarks:", samples)

                    # Optionally: save this face's landmarks to a JSON file
                    # uncomment to save per-frame (will overwrite each frame)
                    # with open(f"face_{fi}_landmarks.json","w") as f:
                    #     json.dump(lm_list, f, indent=2)

            # FPS counter
            frame_count += 1
            if frame_count % 10 == 0:
                now = time.time()
                fps = 10.0 / (now - prev_time)
                prev_time = now
                cv2.putText(frame, f"FPS: {int(fps)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.imshow("MediaPipe FaceMesh", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == ord('s') and results.multi_face_landmarks:
                # save landmarks of first face to CSV and JSON
                lm_list = []
                h, w, _ = frame.shape
                for idx, lm in enumerate(results.multi_face_landmarks[0].landmark):
                    x_px, y_px = normalized_to_pixel_coords(lm.x, lm.y, w, h)
                    lm_list.append((idx, lm.x, lm.y, lm.z, x_px, y_px))
                # save CSV
                with open("face_landmarks.csv", "w", newline="") as csvf:
                    writer = csv.writer(csvf)
                    writer.writerow(["index","x_norm","y_norm","z_norm","x_px","y_px"])
                    writer.writerows(lm_list)
                # save JSON
                with open("face_landmarks.json", "w") as jf:
                    json.dump([{
                        "index": r[0],"x_norm":r[1],"y_norm":r[2],"z_norm":r[3],"x_px":r[4],"y_px":r[5]
                    } for r in lm_list], jf, indent=2)
                print("Saved face_landmarks.csv and .json")

    cap.release()
    cv2.destroyAllWindows()

def run_image(image_path, max_num_faces=2):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h, w, _ = img.shape

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=max_num_faces,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            print("No faces found.")
            return

        for fi, face_landmarks in enumerate(results.multi_face_landmarks):
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=0)
            )
        cv2.imwrite("annotated_image.jpg", img)
        print("Saved annotated_image.jpg")

if __name__ == "__main__":
    # run_webcam(max_num_faces=2)
    # or process an image:
    # run_image("person.jpg", max_num_faces=3)

    # Default: run webcam with up to 1 face
    run_webcam(max_num_faces=1)