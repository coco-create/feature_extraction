# feature_extraction
对视频/图像进行不同等级的特征提取和匹配


### 低级特征：edge pixel area
这三个特征都是对于像素级别的操作，**无法提取特征向量**，只适合用于过滤帧，无法算两帧之间的距离，故不考虑



### 高级特征：
采取算法对图片提取出特征点（关键点 keypoint）， 对特征点得到一个描述符（特征向量），再对特征向量算（欧式/余弦/汉明）距。
采用了ORB，BRIEF，SIFT三个特征，因为HOG **提取时间过长** 不考虑

**差帧:** 为了消除背景的影响，采取差帧处理 （取视频的第一帧作为背景）

#### ORB：
- 原图提取:
![image](https://github.com/coco-create/feature_extraction/blob/master/output/ORB/%E5%8E%9F%E5%9B%BE%E6%8F%90%E5%8F%96.png)
- 差帧处理再提取
![image](https://github.com/coco-create/feature_extraction/blob/master/output/ORB/%E5%B7%AE%E5%B8%A7%E6%8F%90%E5%8F%96.png)


#### BRIEF：
- 原图提取：
![image](https://github.com/coco-create/feature_extraction/blob/master/output/BRIEF/%E5%8E%9F%E5%9B%BE%E5%A4%84%E7%90%86.jpg)
- 差帧处理再提取
![image](https://github.com/coco-create/feature_extraction/blob/master/output/BRIEF/%E5%B7%AE%E5%B8%A7%E5%A4%84%E7%90%86.png)

#### SIFT：
- 原图提取:
![image](https://github.com/coco-create/feature_extraction/blob/master/output/SIFT/%E6%95%B0%E6%8D%AE%E9%9B%86%E8%A7%86%E9%A2%91/%E5%8E%9F%E5%9B%BE%E6%8F%90%E5%8F%96.png)
- 差帧处理再提取:
![image](https://github.com/coco-create/feature_extraction/blob/master/output/SIFT/%E6%95%B0%E6%8D%AE%E9%9B%86%E8%A7%86%E9%A2%91/%E5%B7%AE%E5%B8%A7%E5%A4%84%E7%90%86.png)
