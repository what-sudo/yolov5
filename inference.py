import torch
import cv2

model = torch.hub.load('./', 'custom', path='yolov5s.pt', source='local')

windowName = "video"
cap = cv2.VideoCapture(2)

if cap.isOpened():
    while True:
        ret, frame = cap.read()
        if not ret: break

        results = model(frame)
        row,col = results.pandas().xyxy[0].shape

        for i in range(row):
            x1, y1, x2, y2, confidence, cls, name =  results.pandas().xyxy[0].iloc[i]
            cv2.putText(frame, name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            cv2.rectangle(frame, (int(x1), int(y1) - 20), (int(x2), int(y2)), (0, 255, 255), 2, lineType=cv2.LINE_8)

        cv2.imshow(windowName, frame)
        if cv2.waitKey(10) == ord('q'):
            break
else:
    print("Error: VideoCapture failed")

cv2.destroyAllWindows()
