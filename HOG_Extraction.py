import time
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import xlwt


book = xlwt.Workbook()
sheet = book.add_sheet(u'time',cell_overwrite_ok=True)


'''源码处理
class Hog_descriptor():
    #---------------------------#
    #   初始化
    #   cell_size每个细胞单元的像素数
    #   bin_size表示把360分为多少边
    #---------------------------#
    def __init__(self, img, cell_size=16, bin_size=8):
        self.img = img
        self.img = np.sqrt(img / np.max(img))
        self.img = img * 255
        self.cell_size = cell_size
        self.bin_size = bin_size #每个bin的角度大小
        self.angle_unit = 360 / self.bin_size  #bin的个数
    #---------------------------#
    #   获取hog向量和图片
    #---------------------------#
    def extract(self):
        # 获得原图的shape
        height, width = self.img.shape
        # 计算原图的梯度大小
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)

        # cell_gradient_vector用来保存每个细胞的梯度向量
        # height_cell=height / self.cell_size;width_cell=width / self.cell_size
        cell_gradient_vector = np.zeros((int(height / self.cell_size), int(width / self.cell_size), self.bin_size))
        height_cell,width_cell,_ = np.shape(cell_gradient_vector)
        #---------------------------#
        #   计算每个细胞的梯度直方图
        #---------------------------#
        for i in range(height_cell):
            for j in range(width_cell):
                # 获取这个细胞内的梯度大小
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                # 获得这个细胞内的角度大小
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                # 转化为梯度直方图格式
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        # hog图像
        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)
        hog_vector = []
        # block为2x2
        for i in range(height_cell - 1):
            for j in range(width_cell - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))#对向量的每个维度求平方和再开方求向量模
                magnitude = mag(block_vector)
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    #注意lambda函数的使用
                    block_vector = normalize(block_vector, magnitude)
                hog_vector.append(block_vector)
        return hog_vector, hog_image
    #---------------------------#
    #   计算原图的梯度大小
    #   角度大小
    #---------------------------#
    def global_gradient(self):
        # 梯度滤波器，或者说高通滤波器,求一阶或二阶导数
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        #将两张图片可以通过cv2.addWeighted( )按权重进行融合
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        #返回tan=y/x时对应的角度，angleInDegrees选择输出为角度/弧度
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        return gradient_magnitude, gradient_angle
    #---------------------------#
    #   分解角度信息到
    #   不同角度的直方图上
    #---------------------------#
    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                #？？？？？？这是在干啥
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers
    #---------------------------#
    #   计算每个像素点所属的角度
    #---------------------------#
    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        return idx, (idx + 1) % self.bin_size, mod
    #---------------------------#
    #   将梯度直方图进行绘图
    #---------------------------#
    def render_gradient(self, image, cell_gradient):
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image

img = cv2.imread('images/pic3.jpg', cv2.IMREAD_GRAYSCALE)

cell_size = 8

for i in range(1,10):
    bin_size = i

    start = time.time()
    hog = Hog_descriptor(img, cell_size=cell_size, bin_size=bin_size)
    vector, image = hog.extract()
    end = time.time()
    HOG_time = end - start
    print(HOG_time)
    #print(vector)
    #vector.shape()

    plt.imshow(image, cmap=plt.cm.gray)
    plt.title('cell_size=' + str(cell_size) + ',bin_size=' + str(bin_size) + ',t=' + str(HOG_time) + 's')
    # plt.imshow(img, cmap=plt.cm.gray)
    plt.savefig('img_output/fig8-' + str(i) + '.png')
    plt.show()
'''


from skimage import feature as ft

orit = 1
cell_size = 10
block_size = 2


img = cv2.imread('images/pic3.jpg', cv2.IMREAD_GRAYSCALE)

#ft.hog 参数含义
#orientations:int, optional
#Number of orientation bins. 方向梯度的个数
#pixels_per_cell:2-tuple (int, int), optional
#Size (in pixels) of a cell. 每个cell中的像素个数，也决定cell的个数
#cells_per_block:2-tuple (int, int), optional
#Number of cells in each block. 改变后貌似无明显变化？？？

for i in range(3, 8):
    for j in range(4, 20):
        start = time.time()
        features = ft.hog(img, orientations=i, pixels_per_cell=[j, j], cells_per_block=[block_size, block_size],
                          visualize=True)
        end = time.time()

        HOG_time = end - start
        print(HOG_time)
        sheet.write(j, i, HOG_time)

        plt.imshow(features[1], cmap=plt.cm.gray)

        # plt.imshow(img)
        plt.title('orientations= ' + str(i) + 'pixels_per_cell=' + str(j) + 'cells_per_block=' + str(block_size))
        plt.savefig('output/HOG2/fig' + str(i) + '-' + str(j) +'.png')
        plt.show()


book.save('HOG_time.xls')

