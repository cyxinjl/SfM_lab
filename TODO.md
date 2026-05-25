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
    ...
```

### 状态：已完成 (2026-05-25)
- [x] 删除 `compute_sampson_errors()`
- [x] 删除 `compute_ransac_iterations()`
- [x] 删除自定义 `estimate_fundamental_matrix()`
- [x] 新增 `estimate_fundamental_matrix_opencv_ransac()`，使用 `cv2.FM_RANSAC`
- [x] 更新 `main()` 中的调用代码
- [x] 删除重复的 `from_homogeneous()` 定义
- [x] 删除死代码第一个 `camera_intrinsics` 定义
- [x] 修复 `camera_poses[init_j]` 未设置的 bug
- [x] 移除无用的 `math`、`random` 导入
