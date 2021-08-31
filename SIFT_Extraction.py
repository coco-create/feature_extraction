import numpy as np
import cv2
import time
import os
import xlwt


book = xlwt.Workbook()
sheet = book.add_sheet(u'time',cell_overwrite_ok=True)


def load_image(path, gray=False):
    if gray:
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        return cv2.imread(path)

def transform(origin):
    h, w, _ = origin.shape
    generate_img = np.zeros(origin.shape)
    for i in range(h):
        for j in range(w):
            generate_img[i, w - 1 - j] = origin[i, j]
    return generate_img.astype(np.uint8)


def sift(img,threshold):
    img1 = load_image(img)
    img2 = transform(img1)

    #合并img和img对称转换之后的图片
    combine1 = np.hstack((img1, img2))

    # 实例化:
    sift = cv2.xfeatures2d.SIFT_create()

    # 计算关键点和描述子
    # 其中kp为关键点keypoints
    # des为描述子descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # print(des1)
    # print(des2)

    # 绘出关键点
    # 其中参数分别是源图像、关键点、输出图像、显示颜色
    img3 = cv2.drawKeypoints(img1, kp1, img1, color=(0, 255, 255))
    img4 = cv2.drawKeypoints(img2, kp2, img2, color=(0, 255, 255))

    combine2 = np.hstack((img3, img4))

    # 参数设计和实例化
    index_params = dict(algorithm=1, trees=6)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 利用knn计算两个描述子的匹配
    matche = flann.knnMatch(des1, des2, k=2)
    matchesMask = [[0, 0] for i in range(len(matche))]

    # 绘出匹配效果
    result = []
    for m, n in matche:
        if m.distance < threshold * n.distance:
            result.append([m])

    img5 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matche, None, flags=2)
    img6 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, result, None, flags=2)

    #cv2.imshow("MatchResult", img5)

    return combine1, combine2, img5, img6


def main():
    path = 'images/'
    folders = os.listdir(path)
    print(folders)


    #satre = time.time()
    for i in range(10):
        # sift('images/pic3.jpg')
        j = 0
        for img_path in folders:
            img = path + img_path
            # print(img_path)
            # sift(img_path)
            start = time.time()
            img_combine1, img_combine2, img_without_threshold, img_with_threshold = sift(img,0.8)
            end = time.time()
            all_time = end - start

            print(all_time)
            sheet.write(i, j, all_time)
            j += 1

            cv2.imshow("img_combine1", img_combine1)
            # cv2.waitKey(0)
            cv2.imwrite('output/SIFT/img_combine1' + img_path, img_combine1)
            cv2.imshow("img_combine2", img_combine2)
            # cv2.waitKey(0)
            cv2.imwrite('output/SIFT/img_combine2' + img_path, img_combine2)
            cv2.imshow("img_without_threshold", img_without_threshold)
            cv2.imwrite('output/SIFT/img_without_threshold' + img_path, img_without_threshold)
            cv2.imshow("img_with_threshold", img_with_threshold)
            cv2.imwrite('output/SIFT/img_with_threshold' + img_path, img_with_threshold)
            cv2.waitKey(0)



    book.save('SIFT_time.xls')







'''
        img_combine1, img_combine2, img_tran = sift(img_path)
        cv2.imshow("img_combine1", img_combine1)
        cv2.waitKey(0)
        cv2.imshow("img_combine2", img_combine2)
        cv2.waitKey(0)
        cv2.imshow("img_match", img_tran)
        cv2.waitKey(0)
        
'''


'''

img1 = load_image('images/QQ_2.jpg' )
img2 = load_image('images/QQ_1.jpg' )

print(type(img1))
print(type(img2))



combine1 = np.hstack((img1, img2))

# 实例化
sift = cv2.xfeatures2d.SIFT_create()

# 计算关键点和描述子
# 其中kp为关键点keypoints
# des为描述子descriptors 是一个维的向量
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)


print(des1)
print(des2)

# 绘出关键点
# 其中参数分别是源图像、关键点、输出图像、显示颜色
img3 = cv2.drawKeypoints(img1, kp1, img1, color=(0, 255, 255))
img4 = cv2.drawKeypoints(img2, kp2, img2, color=(0, 255, 255))

combine2 = np.hstack((img3, img4))

# 参数设计和实例化
index_params = dict(algorithm=1, trees=6)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# 利用knn计算两个描述子的匹配
matche = flann.knnMatch(des1, des2, k=2)
#matchesMask = [[0, 0] for i in range(len(matche))]
#print(matchesMask)

threshold = 0.4

# 绘出匹配效果
result = []
for m, n in matche:
    if m.distance < threshold * n.distance:
        result.append([m])

img5 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matche, None, flags=2)
img6 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, result, None, flags=2)



cv2.imshow("img1", combine1)
cv2.waitKey(0)
cv2.imshow("img2", combine2)
cv2.waitKey(0)
cv2.imshow("MatchResult_without_threshold", img5)
cv2.waitKey(0)
cv2.imshow("MatchResult_with_threshold = "+ str(threshold), img6)
cv2.waitKey(0)


'''

if __name__ == '__main__':
    main()