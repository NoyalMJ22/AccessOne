import cv2
import joblib
from features import get_features

model = joblib.load("model.pkl")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    eye_dist, dwell, blink = get_features(frame)

    prediction = model.predict([[eye_dist, dwell, blink]])

    if prediction == "LOOK":
        cv2.putText(frame, "SELECT", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    if prediction == "LONG_BLINK":
        cv2.putText(frame, "EMERGENCY", (50,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Live", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()