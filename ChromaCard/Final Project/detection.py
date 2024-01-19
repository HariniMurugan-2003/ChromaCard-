import cv2
from flask import Flask, render_template, request, redirect, Response, send_from_directory
from ultralytics import YOLO

app = Flask(__name__)
# Placeholder for the password (you can implement a more secure solution)
PASSWORD = "1234"

# OpenCV VideoCapture object to access the camera
camera = cv2.VideoCapture(0)  # Use '0' for the default camera, change to other numbers for different cameras if available

def generate_frames():
    model = YOLO("yolov8s.pt")
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert the frame to JPEG format
            result = model.predict(frame)
            for box in result[0].boxes.xyxy.tolist():
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("home.html")





@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)