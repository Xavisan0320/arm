import cv2
import os

save_dir = "dataset/images"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(1)

count = 0

print("SPACE: 撮影")
print("ESC  : 終了")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)

    if key == 32:   # Space
        filename = os.path.join(save_dir, f"img_{count:04d}.jpg")
        cv2.imwrite(filename, frame)
        print(filename)
        count += 1

    elif key == 27: # ESC
        break

cap.release()
cv2.destroyAllWindows()