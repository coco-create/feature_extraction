import cv2
import numpy as np
import time
import xlwt

book = xlwt.Workbook()
sheet = book.add_sheet(u'time',cell_overwrite_ok=True)

camera = cv2.VideoCapture(0)
gray = None
i = 0

while True:
    ret, frame = camera.read()
    if not ret:
        break

    start = time.time()

    firstframe = gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if firstframe is None:
        firstframe = gray
        continue

    frameDelta = cv2.absdiff(firstframe, gray)#两幅图的差的绝对值输出到另一幅图上面来
    thresh = cv2.threshold(frameDelta, 50, 255, cv2.THRESH_BINARY)[1]
    #给两幅图的差异设定阈值
    thresh = cv2.dilate(thresh, None, iterations=2)
    #dilate函数可以对输入图像用特定结构元素进行膨胀操作，该结构元素确定膨胀操作过程中的邻域的形状，各点像素值将被替换为对应邻域上的最大值
    # cnts= cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(thresh)
    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    end = time.time()
    area_time = end - start

    ratio = w * h / (640 * 480) * 100
    print('时间为'+ str(area_time)+'所占比例为'+ str(ratio)+'%')

    if ratio:
        sheet.write(i, 0, area_time)
        sheet.write(i, 1, ratio/100)
        i += 1

    cv2.imshow("frame", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("frame2", frameDelta)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

book.save('area_time.xls')

camera.release()
cv2.destroyAllWindows()