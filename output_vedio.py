import  cv2
#读视频
capture = cv2.VideoCapture("C:\\Users\\老万选教授\\Desktop\\数据源-视频\\View_001.mp4")
#写视频
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = capture.get(cv2.CAP_PROP_FPS)
size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
vedio_out = cv2.VideoWriter('camera_out.avi', fourcc, fps, size)

print('1111')
#按帧 读视频流
if capture.isOpened():
    while True:
        ret, frame = capture.read()#
        if not ret:
            break

        vedio_out.write(frame)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

capture.release()
