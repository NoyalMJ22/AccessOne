import cv2
import csv
from features import get_features

cap = cv2.VideoCapture(0)

file = open("dataset.csv", "a", newline='')
writer = csv.writer(file)

print("Press:")
print("1 → LOOK")
print("2 → LONG BLINK")
print("3 → RANDOM")

while True:
    ret, frame = cap.read()

    eye_dist, dwell, blink = get_features(frame)

    key = cv2.waitKey(1)

    if key == ord('1'):
        writer.writerow([eye_dist, dwell, blink, "LOOK"])
        print("LOOK saved")

    if key == ord('2'):
        writer.writerow([eye_dist, dwell, blink, "LONG_BLINK"])
        print("LONG BLINK saved")

    if key == ord('3'):
        writer.writerow([eye_dist, dwell, blink, "RANDOM"])
        print("RANDOM saved")

    cv2.imshow("Recording", frame)

    if key == 27:
        break

cap.release()
file.close()
cv2.destroyAllWindows()