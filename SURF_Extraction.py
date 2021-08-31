import cv2
import os

path = 'video/'
folders = os.listdir(path)
print(folders)

i = 0
for img_path in folders:
    img = path + img_path
    print(img)
    capture = cv2.VideoCapture(img)

    i += 1
    # get the first frame
    ret, frame = capture.read()
    # show a frame
    cv2.imshow("capture", frame)
    cv2.imwrite("FirstFrame/" + str(i) + '.jpg',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()


