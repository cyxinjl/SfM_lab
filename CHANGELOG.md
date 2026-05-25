# CHANGELOG

> 以下所有修改均由 AI Agent (Alan) 执行，时间：2026-05-25

## 一、删除

| 项目 | 说明 |
|------|------|
| `compute_sampson_errors()` | 自定义 Sampson 误差计算，~16 行 |
| `compute_ransac_iterations()` | 自定义 RANSAC 自适应迭代次数计算，~23 行 |
| `estimate_fundamental_matrix()` | 自定义 RANSAC 八点法 F 矩阵估计，~112 行 |
| 重复的 `from_homogeneous()` | 第二处定义（原约第 867 行），只处理 2D 数组，不如第一个版本完善 |
| 死代码 `camera_intrinsics` | 第一处定义（原约第 2109 行），键名与实际使用不匹配，从未生效 |
| `import math` | 不再使用 |
| `import random` | 不再使用 |

## 二、新增

| 项目 | 说明 |
|------|------|
| `estimate_fundamental_matrix_opencv_ransac()` | 使用 `cv2.findFundamentalMat(method=cv2.FM_RANSAC)` 替代手写 RANSAC，代码量从 ~110 行缩减至 ~45 行 |
| `max_optimized_points` 参数 | 在 `run_bundle_adjustment_fixed_K` 和 `run_bundle_adjustment_refine_focal` 中新增，默认 300，BA 前对三维点随机下采样 |
| `verbose=2` | BA 中 `least_squares` 调用改为 `verbose=2`，终端可看到 scipy 优化进度 |

## 三、修复

| 项目 | 说明 |
|------|------|
| `camera_poses[init_j]` 未写入 | 初始图像对第二张图像的位姿（R_init, t_init）此前只声明未存入 `camera_poses`，导致增量 SfM 无法以该图像为参考做三角化 |
| BA 程序假死 | 原因为 scipy 有限差分雅可比随 N_params 剧烈膨胀。修复方案：每次 BA 前随机下采样三维点至 300 个以内，将 N_params 控制在 ~1000 级别 |

## 四、参数调整

| 参数 | 旧值 | 新值 | 位置 |
|------|------|------|------|
| BA 每轮 `max_nfev` | 3 | 10 | `incremental_sfm_expansion` |
| BA 周期性 `max_nfev` | 3 | 30 | `incremental_sfm_expansion` |
| BA 最终 `max_nfev` | 3 | 100 | `incremental_sfm_expansion` |
| BA `max_optimized_points` | 无 | 300 | `run_bundle_adjustment_fixed_K`、`run_bundle_adjustment_refine_focal` |
| `min_pose_inliers` | 30 | 20 | `evaluate_initial_pair`、`select_initial_pair`、`main()` |
| `min_pnp_points` | 20 | 15 | `register_image_by_pnp`、`main()` |
| `min_common_points` | 20 | 15 | `select_next_edge_for_pnp`、`main()` |
| `reproj_error_threshold` | 5.0 px | 8.0 px | `is_triangulated_point_valid`、`triangulate_new_tracks_after_registering_image` |

## 五、BA 架构变更

### 5.1 三级 BA 梯度

| 层级 | `max_nfev` | 用途 |
|------|-----------|------|
| 每轮轻量 BA | 10 | 维持结构，几乎无开销 |
| 周期性中等 BA（每 5 轮） | 30 | 适度优化，区别于每轮 BA |
| 最终全局 BA | 100 | 充分优化 |

### 5.2 三维点下采样

- 每次 BA 前，若 `points3D` 超过 `max_optimized_points`（默认 300），用固定种子（42）随机采样至 300 个点
- N_params ≈ 可变相机×6 + 300×3 ≈ 1000，有限差分雅可比约 2000 次函数评估
- 每次函数评估开销取决于观测数量，个人电脑上可在数秒内完成
- 不同 BA 调用采样不同的随机子集，多次调用覆盖全部点

### 5.3 终端可见进度

- BA 开始时打印参数规模：cameras、points、params、observations、max_nfev
- `least_squares(verbose=2)` 输出每次迭代的 cost、gradient norm 等信息

---

*以上修改由 AI Agent (Alan) 执行。*
