
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
import json

app = FastAPI()

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Load dress dataset
DRESS_DATA_FILE = "dress_data.json"
try:
    with open(DRESS_DATA_FILE, "r") as f:
        dress_dataset = json.load(f)
except FileNotFoundError:
    dress_dataset = [
        {
            "name": "Casual Red Dress",
            "image_url": "https://example.com/red-dress.jpg ",
            "attributes": {"color": "red", "style": "casual", "size": "M"},
            "compatibility": {"hip": [34, 40], "waist": [26, 32]}
        }
    ]
    with open(DRESS_DATA_FILE, "w") as f:
        json.dump(dress_dataset, f, indent=4)

def get_body_measurements(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Could not load image"}
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return {"error": "No body detected"}

    landmarks = results.pose_landmarks.landmark

    # Hip width
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    hip_width_pixels = abs(left_hip.x - right_hip.x) * image.shape[1]
    hip_inches = round(hip_width_pixels * 0.0393701, 2)

    # Waist width
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    waist_width_pixels = abs(left_shoulder.x - right_shoulder.x) * image.shape[1]
    waist_inches = round(waist_width_pixels * 0.0393701, 2)

    return {
        "hip": f"{hip_inches} inches",
        "waist": f"{waist_inches} inches"
    }

@app.post("/recommend-dresses-from-photo")
async def recommend_dresses_from_photo(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return {"error": "Only image files allowed"}

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Save uploaded image
    file_location = f"/content/{file.filename}"
    cv2.imwrite(file_location, img)

    # Get body measurements
    measurements = get_body_measurements(file_location)
    if "error" in measurements:
        return {"error": measurements["error"]}

    # Detect skin tone
    try:
        result = DeepFace.analyze(img_path=file_location, actions=['race'], enforce_detection=False)
        skin_tone = result.get("dominant_race", "N/A")
    except Exception as e:
        return {"error": f"Skin tone detection failed: {str(e)}"}

    # Match with dress dataset
    recommended = []

    for dress in dress_dataset:
        comp = dress.get("compatibility", {})
        hip_range = comp.get("hip", [])
        waist_range = comp.get("waist", [])

        hip_match = hip_range and hip_range[0] <= measurements["hip"] <= hip_range[1]
        waist_match = waist_range and waist_range[0] <= measurements["waist"] <= waist_range[1]

        if hip_match and waist_match:
            recommended.append(dress)

    return {
        "measurements": measurements,
        "skin_tone": skin_tone,
        "recommended_dresses": recommended[:10],
        "note": "You can improve color matching and add virtual try-on next"
    }

@app.get("/")
def read_root():
    return {"message": "Hello from your fashion API!"}
    