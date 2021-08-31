import cv2
import time
import xlwt
import os
from skimage import data,filters,img_as_ubyte

book = xlwt.Workbook()
sheet = book.add_sheet(u'time',cell_overwrite_ok=True)


path = 'video/'
folders = os.listdir(path)


j = 0
for img_path in folders:
    img = path + img_path
    print(img)
    capture = cv2.VideoCapture(img)

    firstframe = None
    i = 1
    if capture.isOpened():
        while True:
            ret, frame = capture.read()  # 如果要读取实时视频 则将此处的capture 改成 camera即可
            if not ret:
                break

            start = time.time()
            # 作差帧后再提取
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if firstframe is None:
                firstframe = gray
                continue
            frameDelta = cv2.absdiff(firstframe, gray)  # 两幅图的差的绝对值输出到另一幅图上面来

            # sobel边缘检测
            edges = cv2.Sobel(frame, cv2.CV_16S, 1, 1)
            # 浮点型转成uint8型
            edges = cv2.convertScaleAbs(edges)
            '''sobel算子有两种方法计算（opencv和skimage）下面的是skimage，效果欠佳
            # sobel边缘检测
            edges = filters.sobel(frame)
            # 浮点型转成uint8型
            edges = img_as_ubyte(edges)
            '''
            end = time.time()
            edge_time = end - start
            print(edge_time)

            cv2.imshow("frame", frame)
            cv2.imshow("edges", edges)
            cv2.imshow("frameDelta", frameDelta)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if edge_time:
                sheet.write(i, j, edge_time)
                i += 1
    j += 1

book.save('edge_time.xls')

capture.release()
cv2.destroyAllWindows()