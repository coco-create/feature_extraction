img = mpimg.imread("E:/FFOutput/1024x768_188k.jpg")
img = img[:, :, :]/255
print("原始图像尺寸：", img.shape)

# 灰度化：灰度化是一个加权平均的过程！
a1, a2, a3 = 0.2989, 0.5870, 0.1140
img_gray = img[:,:,0]*a1+img[:,:,1]*a2+img[:,:,2]*a3
print("灰度化后尺寸：", img_gray.shape)

# 奇异值分解，用于压缩图像
M, Q, N = np.linalg.svd(img_gray, full_matrices=True)
#Q=np.diag(Q) #对角化处理

# re_img=M[:, 0:50] @ Q[0:50, 0:50] @ N[0:50, :]  # 恢复图像
# plt.imshow(np.real(re_img), cmap=plt.get_cmap("gray"))
# plt.show()
img1 = mpimg.imread("E:/FFOutput/1.png")
img1 = img1[:, :, :]/255
img_gray1 = img1[:,:,0]*a1+img1[:,:,1]*a2+img1[:,:,2]*a3
M, Q1, N = np.linalg.svd(img_gray1, full_matrices=True)
ans=0
for i in range(min(len(Q),len(Q1))):
    if (Q1[i] == 0):
        continue;
    print(Q[i],Q1[i],Q==Q1)
    p = Q[i] / Q1[i]
    ans += p * np.log2(p)

print('\033[1;31m-----------------------------------------------\033[0m')
print('\033[1;31m Entropy =', abs(ans), '\033[0m')