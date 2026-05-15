# 可以进行的修改

## SIFT算法代码

### 预处理阶段
利用 RANSAC 方式计算F矩阵，将代码直接修改成 OpenCV 库中含有的。下面是需要删掉的代码
```python
compute_sampson_errors()
compute_ransac_iterations()
estimate_fundamental_matrix(kp1, kp2, matches, threshold=1.0, confidence=0.99, initial_max_iterations=50000, min_iterations=100)
```

下面是需要修改的函数内容
```python
def estimate_fundamental_matrix_opencv_ransac(kp1, kp2, matches, ransac_threshold=3.0, confidence=0.99, max_iters=50000): # 用于估计基础矩阵 
    """
    使用 OpenCV 自带 RANSAC 估计基础矩阵 F，并筛选几何内点。

    参数：
    kp1:第一张图像的 OpenCV keypoints。
    kp2:第二张图像的 OpenCV keypoints。
    matches:两张图像之间的匹配结果，元素类型为 cv2.DMatch。
    ransac_threshold: RANSAC 内点判断阈值，单位近似为像素。
    confidence: RANSAC 置信度。
    max_iters:最大迭代次数。

    返回：
    F:基础矩阵，3×3。
    inlier_matches:通过 RANSAC 几何验证的匹配点。
    mask:与 matches 等长的一维数组。1 表示内点，0 表示外点。
    """
    if matches is None or len(matches) < 8:
        return None, [], None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    if pts1.shape[0] < 8 or pts2.shape[0] < 8:
        return None, [], None
    try:
        F, mask = cv2.findFundamentalMat(
            pts1,
            pts2,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=ransac_threshold,
            confidence=confidence,
            maxIters=max_iters)
    except cv2.error as e:
        print("cv2.findFundamentalMat 出错：")
        print(e)
        return None, [], None
    
    if F is None or mask is None:
        return None, [], None
    if F.shape != (3, 3):
        return None, [], None

    mask = mask.ravel().astype(np.uint8)
    inlier_matches = [m for m, keep in zip(matches, mask) if keep == 1]

    return F, inlier_matches, mask
```

###