import cv2
import numpy as np
import math
from pathlib import Path
import random
from collections import defaultdict
from scipy.optimize import least_squares

def create_folder(folder_path):                     #负责创建用于存储结果的文件夹
    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)

def get_image_files(image_folder):                  #负责获取图片文件列表，并按文件名排序
    image_folder = Path(image_folder)
    supported_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    image_files = [
        file for file in image_folder.iterdir()
        if file.suffix.lower() in supported_extensions
    ]                                               #遍历获得上面需要的图像文件
    image_files = sorted(image_files)               #对图片对象进行排序，确保处理顺序一致

    return image_files                              #返回一个包含所有图片路径的列表，供后续处理使用

def extract_sift_features(image_path, sift):        #提取 SIFT 特征的函数，输出原图、灰度图、关键点和描述子
    image = cv2.imread(str(image_path))

    if image is None:                               #无法读取的报错情况
        print(f"无法读取图片：{image_path}")
        return None, None, None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #将图像转换为灰度图，将3个颜色通道转换为1个通道，SIFT 只需要灰度图进行特征检测和描述子计算
    keypoints, descriptors = sift.detectAndCompute(gray, None)  #使用 SIFT 对象的 detectAndCompute 方法，输入灰度图，输出关键点和描述子。关键点是一个列表，每个元素包含特征点的位置、尺度、方向等信息；描述子是一个二维数组，每行对应一个关键点的特征向量。

    return image, gray, keypoints, descriptors

def match_two_images(desc1, desc2):         #对两张图的SIFT描述子进行匹配的函数，输入两张图的描述子，输出初步匹配结果列表
    if desc1 is None or desc2 is None:
        return []
    if len(desc1) < 2 or len(desc2) < 2:
        return []

    bf = cv2.BFMatcher(cv2.NORM_L2)         #创建匹配器对象，并用欧式距离衡量
    # knnMatch 中 k=2，表示对每个描述子找两个最相近的候选匹配
    matches = bf.knnMatch(desc1, desc2, k=2) #找出最优和次优匹配
    prime_matches = []
    ratio_threshold = 0.75 # SIFT 匹配的距离比率阈值，越小表示匹配越严格
    # 如果最近邻明显优于次近邻，认为匹配可靠
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            prime_matches.append(m)

    return prime_matches 

def to_homogeneous(points): # 将欧式坐标转化为齐次坐标
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        return np.append(points, 1.0)
    elif points.ndim == 2:
        ones = np.ones((points.shape[0], 1), dtype=np.float64)
        return np.hstack((points, ones))
    else:
        raise ValueError("输入点的维度必须为1或2。")
    
def from_homogeneous(points_h, eps=1e-8): # 将齐次坐标转化为欧式坐标
    #如果 w 或 W 接近 0，则对应结果返回 NaN。
    points_h = np.asarray(points_h, dtype=np.float64)
    if points_h.ndim == 1:
        w = points_h[-1]
        if abs(w) <= eps:
            return np.full(points_h.shape[0] - 1, np.nan, dtype=np.float64)
        return points_h[:-1] / w

    elif points_h.ndim == 2:
        w = points_h[:, -1]
        points = np.full((points_h.shape[0], points_h.shape[1] - 1), np.nan, dtype=np.float64)
        valid = np.abs(w) > eps
        points[valid] = points_h[valid, :-1] / w[valid, None]
        return points
            
    else:
        raise ValueError("输入只能是一维或二维 numpy 数组。")

def compute_sampson_errors(pts1, pts2, F): # 计算匹配点相对于基础矩阵 F 的误差距离。
    # pst1表示第一张图的匹配点坐标，pts2表示第二张图的匹配点坐标
    pts1 = np.asarray(pts1, dtype=np.float64) 
    pts2 = np.asarray(pts2, dtype=np.float64)
    pts1_h = to_homogeneous(pts1) # 将匹配点坐标转换为齐次坐标
    pts2_h = to_homogeneous(pts2) 
    F = np.asarray(F, dtype=np.float64)
    num_points = pts1.shape[0] #匹配点的数量
    
    F_pts1 = F @ pts1_h.T # F x1
    Ft_pts2 = F.T @ pts2_h.T # F^T x2
    numerator = np.sum(pts2_h * (F @ pts1_h.T).T, axis=1) ** 2 # (x2^T F x1)^2，表示匹配点在基础矩阵 F 上的残差平方
    denominator = (F_pts1[0, :] ** 2 + F_pts1[1, :] ** 2 + Ft_pts2[0, :] ** 2 + Ft_pts2[1, :] ** 2 ) #用在分母上进行归一化处理，得到真正的像素几何
    eps = 1e-12 #为了避免除以零的情况，
    errors = numerator / (denominator + eps) 
    return errors

def compute_ransac_iterations(inlier_ratio, sample_size, confidence, max_iterations): # 根据当前内点比例，自适应计算 RANSAC 所需迭代次数。
    """参数注释：
    inlier_ratio:当前最佳模型的内点比例。
    sample_size:每次估计模型需要采样的点数，在这里基础矩阵八点法中值的大小为8。
    confidence:希望至少采到一次全内点样本的概率。
    max_iterations:当前最大允许迭代次数，要在这个基础上迭代优化。
    
    iterations:根据当前内点比例估计出的迭代次数。
    """
    if inlier_ratio <= 0:
        return max_iterations
    if inlier_ratio >= 1:
        return 1

    prob_all_inliers = inlier_ratio ** sample_size # 一次采样中所有 sample_size 个点都是内点的概率
    eps = 1e-12 # 防止 log(0) 和 log(1) 的情况。
    prob_all_inliers = min(max(prob_all_inliers, eps), 1.0 - eps)

    numerator = math.log(1.0 - confidence)
    denominator = math.log(1.0 - prob_all_inliers)
    iterations = int(math.ceil(numerator / denominator))
    iterations = max(1, min(iterations, max_iterations))
    return iterations

def estimate_fundamental_matrix(kp1, kp2, matches, threshold=1.0, confidence=0.99, initial_max_iterations=50000, min_iterations=100): # RANSAC 算法主函数
    """
    参数：
    kp1:第一张图像的 OpenCV keypoints 列表。
    kp2:第二张图像的 OpenCV keypoints 列表。
    matches:两张图像之间的匹配点列表，元素类型为 cv2.DMatch。
    threshold:Sampson distance 内点判断阈值。
    confidence:RANSAC 置信度。
    initial_max_iterations:初始最大迭代次数。
    min_iterations:最少迭代次数，避免过早停止。
    sample_size:每次采样点数，在这里八点法估计基础矩阵时为 8。

    返回参数：
    best_F:估计得到的基础矩阵 F,形状 3*3。
    inlier_matches: RANSAC 内点匹配。
    inlier_mask:与 matches 等长的一维数组。1 表示内点,0 表示外点。
    info:字典，包含迭代次数、内点数量、内点比例等信息。
    """
    sample_size = 8  # 八点法需要采样的点数
    if matches is None or len(matches) < sample_size:
        info = {
            "success": False,
            "reason": "匹配点数量少于八点法所需的最小点数",
            "iterations_used": 0,
            "best_score": 0,
            "inlier_ratio": 0.0
        }
        return None, [], None, info

    pts1 = np.float64([kp1[m.queryIdx].pt for m in matches]) # 从匹配点列表中提取匹配点坐标，转换为浮点数数组
    pts2 = np.float64([kp2[m.trainIdx].pt for m in matches])
    num_matches = len(matches)
    best_F = None
    best_mask = None # 用于标记内点的布尔数组，长度与 matches 相同，True 表示内点，False 表示外点
    best_score = 0 # 当前最佳模型的内点数量
    best_inlier_ratio = 0.0 # 当前最佳模型的内点比例
    max_iterations = initial_max_iterations # 当前允许的最大迭代次数，会在循环中动态更新
    iteration = 0

    while iteration < max_iterations: # RANSAC 主循环，直到达到最大迭代次数
        iteration += 1
        sample_indices = random.sample(range(num_matches), sample_size) # 从所有匹配点中随机选择 8 对点
        sample_pts1 = pts1[sample_indices]
        sample_pts2 = pts2[sample_indices]

        F_candidate, _ = cv2.findFundamentalMat(sample_pts1, sample_pts2, method=cv2.FM_8POINT) # 使用8点法估计F

        if F_candidate is None:
            continue
        if F_candidate.shape != (3, 3):
            continue
        if not np.isfinite(F_candidate).all(): # 如果 F 中存在 NaN 或 Inf，说明估计失败，跳过本次迭代
            continue

        errors = compute_sampson_errors(pts1, pts2, F_candidate) # 根据 Sampson distance 判断哪些匹配点是内点
        current_mask = errors < threshold
        current_score = int(np.sum(current_mask))
        current_inlier_ratio = current_score / num_matches

        
        if current_score > best_score: # d) 如果当前模型更好，则更新迭代次数
            best_score = current_score
            best_F = F_candidate
            best_mask = current_mask
            best_inlier_ratio = current_inlier_ratio

            # 根据当前最佳内点比例，重新计算所需迭代次数
            estimated_iterations = compute_ransac_iterations(
                inlier_ratio=best_inlier_ratio,
                sample_size=sample_size,
                confidence=confidence,
                max_iterations=max_iterations
            )
            max_iterations = max(min_iterations, estimated_iterations) # 不能低于最小迭代次数

    if best_F is None or best_mask is None:
        info = {
            "success": False,
            "reason": "没有找到有效的基础矩阵",
            "iterations_used": iteration,
            "best_score": 0,
            "inlier_ratio": 0.0,
            "adaptive_max_iterations": max_iterations
        }
        return None, [], None, info

    inlier_pts1 = pts1[best_mask]
    inlier_pts2 = pts2[best_mask]

    if len(inlier_pts1) >= sample_size:
        refined_F, _ = cv2.findFundamentalMat(inlier_pts1, inlier_pts2, method=cv2.FM_8POINT)
        if refined_F is not None and refined_F.shape == (3, 3):
            if np.isfinite(refined_F).all():
                best_F = refined_F
                errors = compute_sampson_errors(pts1, pts2, best_F)
                best_mask = errors < threshold
                best_score = int(np.sum(best_mask))
                best_inlier_ratio = best_score / num_matches
    inlier_mask = best_mask.astype(np.uint8) # 将布尔数组转换为 uint8 类型，1 表示内点，0 表示外点

    inlier_matches = [m for m, keep in zip(matches, inlier_mask) if keep == 1] # 根据内点掩码过滤出内点匹配列表

    info = {
        "success": True,
        "iterations_used": iteration,
        "best_score": best_score,
        "inlier_ratio": best_inlier_ratio,
        "total_matches": num_matches,
        "threshold": threshold,
    }

    return best_F, inlier_matches, inlier_mask, info

class UnionFind: # 用于后续步骤中对匹配点进行 Tracks 构建。
    def __init__(self):
        self.parent = {}

    def find(self, x):
        """
        查找 x 所属集合的代表元素。
        x 的形式为：(image_id, keypoint_id)
        """
        if x not in self.parent:
            self.parent[x] = x

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def union(self, a, b):
        """
        合并 a 和 b 所在的集合。
        """
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a != root_b:
            self.parent[root_b] = root_a

def build_tracks_from_union_find(track_builder): # 将 UnionFind 中的匹配关系整理成 tracks。
    """
    数据形式说明：
    track_builder.parent 是一个字典，键是 (image_id, keypoint_id)，值是该观测点所属集合的代表元素 (root)。
    tracks:list, 每个元素是一个 track。
        track 的形式为：[(image_id, keypoint_id), ...]
    """

    groups = defaultdict(list) # 使用 defaultdict 来自动创建列表，避免 KeyError。当访问一个不存在的键时，会自动创建一个空列表作为该键的值。
    for obs in track_builder.parent.keys(): # 遍历所有观测点，根据它们所属集合的代表元素进行分组
        root = track_builder.find(obs)
        groups[root].append(obs)

    tracks = []
    for root, observations in groups.items():
        tracks.append(observations)

    return tracks

def filter_valid_tracks(tracks, min_track_length = 3): # 用于过滤掉不满足条件的 tracks，确保后续 SfM 处理的稳定性。
    valid_tracks = []
    for track in tracks:
        if len(track) < min_track_length:
            continue 

        image_ids = [obs[0] for obs in track]
        if len(image_ids) != len(set(image_ids)):
            continue

        valid_tracks.append(track)
    return valid_tracks

def build_observation_to_track(valid_tracks): #建立 observation 到 track_id 的映射，方便后续 SfM 处理使用。
    """
    返回参数：
    observation_to_track:
        dict
        key: (image_id, keypoint_id)
        value: track_id
    """

    observation_to_track = {}
    for track_id, track in enumerate(valid_tracks):
        for obs in track:
            observation_to_track[obs] = track_id

    return observation_to_track

def estimate_initial_K_from_image(image, focal_scale=1.2): # 根据图像尺寸估计初始相机内参 K。
    """
    参数: image: OpenCV 读取的图像, focal_scale: 焦距估算系数。
    返回: K: 3*3 相机内参矩阵。
    """

    h, w = image.shape[:2]
    f = focal_scale * max(w, h)
    cx = w / 2.0
    cy = h / 2.0
    K = np.array([
        [f, 0.0, cx],
        [0.0, f, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    return K

def build_K_from_intrinsics(camera_intrinsics):
    """
    根据 camera_intrinsics 字典构造相机内参矩阵 K。
    K:3×3 相机内参矩阵。
    """
    f = camera_intrinsics["f"]
    cx = camera_intrinsics["cx"]
    cy = camera_intrinsics["cy"]

    K = np.array([
        [f,   0.0, cx],
        [0.0, f,   cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    return K

def enforce_essential_matrix_constraints(E):
    """
    对 E 做 SVD 修正，使其更接近合法本质矩阵。本质矩阵需要满足的条件有：
    1. rank(E) = 2
    2. 两个非零奇异值相等
    这里把奇异值修正为 [1, 1, 0]。
    """

    U, S, Vt = np.linalg.svd(E)
    E_fixed = U @ np.diag([1.0, 1.0, 0.0]) @ Vt
    return E_fixed

def recover_pose_from_F(kp1, kp2, inlier_matches, F, K):
    """
    根据基础矩阵 F 和估计内参 K，恢复两张图像之间的相对位姿 R, t。
    
    输入参数：
    kp1, kp2: 两张图像的 keypoints。
    inlier_matches: 经过 F-RANSAC 筛选后的内点匹配。
    F:基础矩阵。
    K:初始相机内参矩阵。
    
    返回参数：
    R:第二个相机相对于第一个相机的旋转矩阵。
    t:第二个相机相对于第一个相机的平移方向。
    pose_inlier_matches: recoverPose 进一步筛选后的内点匹配。
    pose_mask: recoverPose 返回的内点 mask。
    E: 本质矩阵。
    """

    if F is None or inlier_matches is None or len(inlier_matches) < 8:
        return None, None, [], None, None

    pts1 = np.float64([kp1[m.queryIdx].pt for m in inlier_matches]) # 从内点匹配列表中提取匹配点坐标，转换为浮点数数组
    pts2 = np.float64([kp2[m.trainIdx].pt for m in inlier_matches])
    E = K.T @ F @ K 
    E = enforce_essential_matrix_constraints(E)

    try:
        retval, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2,cameraMatrix=K) # 使用 OpenCV 的 recoverPose 函数，根据本质矩阵 E 和匹配点坐标，恢复相对位姿 R 和 t。
    except cv2.error as e:
        print("recoverPose 失败：", e)
        return None, None, [], None, None

    if pose_mask is None:
        return None, None, [], None, E
    pose_mask = pose_mask.ravel() # 将 mask 转换为一维数组，长度与 inlier_matches 相同，1 表示 recoverPose 认为的内点，0 表示外点
    pose_inlier_matches = [m for m, keep in zip(inlier_matches, pose_mask) if keep != 0] # 根据 recoverPose 的内点 mask 进一步筛选出内点匹配列表
    if len(pose_inlier_matches) < 8:
        return None, None, [], pose_mask, E

    return R, t, pose_inlier_matches, pose_mask, E

def build_projection_matrix(K, R, t):
    """
    构造投影矩阵 P = K [R | t]
    """
    Rt = np.hstack([R, t.reshape(3, 1)])
    P = K @ Rt

    return P

def triangulate_initial_pair(kp1, kp2, matches, K, R, t):
    """
    对初始图像对进行三角化。
    第一张图像作为世界坐标系：
        R1 = I
        t1 = 0
    第二张图像：
        R2 = R
        t2 = t

    返回：points_3d: N * 3 三维点。
        valid_matches:与 points_3d 对应的匹配。
    """

    if matches is None or len(matches) < 2:
        return np.empty((0, 3), dtype=np.float64), []
    
    pts1 = np.float64([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float64([kp2[m.trainIdx].pt for m in matches])

    R1 = np.eye(3, dtype=np.float64)
    t1 = np.zeros((3, 1), dtype=np.float64)
    R2 = R
    t2 = t.reshape(3, 1)
    P1 = build_projection_matrix(K, R1, t1)
    P2 = build_projection_matrix(K, R2, t2)
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = from_homogeneous(points_4d.T)

    cam1_points = points_3d
    cam2_points = (R2 @ points_3d.T + t2).T

    valid = (
        np.isfinite(points_3d).all(axis=1)
        & (cam1_points[:, 2] > 0)
        & (cam2_points[:, 2] > 0)
    ) # 只有当三维点坐标是有限的，并且在两台相机前方（z > 0）时，才认为该三维点是有效的。
    valid_points = points_3d[valid] # 根据有效掩码过滤出有效的三维点坐标，得到一个 N_valid * 3 的数组。
    valid_matches = [
        m for m, keep in zip(matches, valid)
        if keep
    ]

    return valid_points, valid_matches

def compute_triangulation_angles(points_3d, R, t):
    """
    计算三角化点对应的两视图观测射线夹角。

    第一相机: C1 = [0, 0, 0]
    第二相机：
        外参为 X_cam2 = R X_world + t
        所以相机中心 C2 = -R^T t
    参数：
    points_3d: N * 3 三维点，位于第一相机坐标系，也就是当前世界坐标系。
    R, t: 第二相机相对于第一相机的位姿。

    返回: angles_deg:每个三维点对应的视差角，单位为度。
    """

    if points_3d is None or len(points_3d) == 0:
        return np.empty((0,), dtype=np.float64)
    t = t.reshape(3, 1) # 确保 t 是列向量，方便后续计算。

    C1 = np.zeros((3,), dtype=np.float64) # 第一相机中心在世界坐标系中的位置，设为原点 [0, 0, 0]。
    C2 = (-R.T @ t).ravel() # 第二相机中心在世界坐标系中的位置
    rays1 = points_3d - C1[None, :] # 从相机中心到三维点的观测射线向量，shape 为 N * 3
    rays2 = points_3d - C2[None, :] # 从第二相机中心到三维点的观测射线向量，shape 为 N * 3
    norm1 = np.linalg.norm(rays1, axis=1) # 计算每个观测射线的长度，得到一个 N 维的数组，表示每个三维点对应的两条观测射线的长度。
    norm2 = np.linalg.norm(rays2, axis=1)

    valid = (norm1 > 1e-12) & (norm2 > 1e-12)  # 只有当两条观测射线的长度都大于一个小阈值时，才认为该三维点的视差角是有效的。避免除以零或计算不稳定的情况。
    angles = np.full((points_3d.shape[0],), np.nan, dtype=np.float64)
    cos_angles = np.sum(rays1[valid] * rays2[valid], axis=1) / (norm1[valid] * norm2[valid]) # 计算两条观测射线之间的夹角余弦值
    cos_angles = np.clip(cos_angles, -1.0, 1.0) # 由于数值误差，cos_angles 可能略微超出 [-1, 1] 的范围，clip 函数将其限制在合法范围内，避免 arccos 出现 NaN。
    angles[valid] = np.degrees(np.arccos(cos_angles))

    return angles

def evaluate_initial_pair(
    pair,
    pair_info,
    all_results,
    min_pose_inliers=30,
    min_triangulated_points=30,
    min_angle_deg=3.0,
    max_angle_deg=70.0,
    focal_scale=1.2
):
    """
    评价一条图像连接边是否适合作为初始图像对。

    参数：
    pair: (i, j)，图像编号对。
    pair_info: pair_infos[(i, j)] 中保存的信息。
    all_results: 图像结果列表

    输出：
    result: 字典，包含是否成功、R、t、K、三角化点、视差角等信息。
    """

    i, j = pair
    result1 = all_results[i]
    result2 = all_results[j]
    img1 = result1["image"]
    kp1 = result1["keypoints"]
    kp2 = result2["keypoints"]
    F = pair_info["F"]
    inlier_matches = pair_info["inlier_matches"]

    if F is None or inlier_matches is None:
        return {
            "success": False,
            "reason": "F 或 inlier_matches 无效",
            "pair": pair
        }
    
    # 1. 初始估计 K
    K = estimate_initial_K_from_image(img1, focal_scale=focal_scale)  

    # 2. 从 F 恢复 R, t
    R, t, pose_inlier_matches, pose_mask, E = recover_pose_from_F(kp1, kp2, inlier_matches, F, K)

    if R is None or t is None:
        return {
            "success": False,
            "reason": "recoverPose 失败",
            "pair": pair
        }
    if len(pose_inlier_matches) < min_pose_inliers:
        return {
            "success": False,
            "reason": "recoverPose 内点数量不足",
            "pair": pair,
            "num_pose_inliers": len(pose_inlier_matches)
        }

    # 3. 三角化
    points_3d, triangulated_matches = triangulate_initial_pair(kp1, kp2, pose_inlier_matches, K, R, t)
    if len(points_3d) < min_triangulated_points:
        return {
            "success": False,
            "reason": "有效三角化点数量不足",
            "pair": pair,
            "num_triangulated": len(points_3d)
        }

    # 4. 计算视差角
    angles_deg = compute_triangulation_angles(points_3d, R, t)
    valid_angles = angles_deg[np.isfinite(angles_deg)]
    if len(valid_angles) == 0:
        return {
            "success": False,
            "reason": "无法计算有效视差角",
            "pair": pair
        }
    median_angle = float(np.median(valid_angles))
    mean_angle = float(np.mean(valid_angles))

    # 5. 判断视差角范围
    if median_angle < min_angle_deg or median_angle > max_angle_deg:
        return {
            "success": False,
            "reason": "视差角不在有效范围内，三角化不稳定",
            "pair": pair,
            "median_angle": median_angle,
            "mean_angle": mean_angle,
            "num_triangulated": len(points_3d)
        }
    
    # 6. 给候选初始对打分
    num_inliers = pair_info.get("num_inliers", len(inlier_matches))
    inlier_ratio = pair_info.get("inlier_ratio", 0.0)

    score = (
        len(points_3d)
        * inlier_ratio
        * min(median_angle / min_angle_deg, 3.0)
    )

    return {
        "success": True,
        "pair": pair,
        "K": K,
        "E": E,
        "R": R,
        "t": t,
        "pose_inlier_matches": pose_inlier_matches,
        "triangulated_matches": triangulated_matches,
        "points_3d": points_3d,
        "angles_deg": angles_deg,
        "median_angle": median_angle,
        "mean_angle": mean_angle,
        "num_inliers": num_inliers,
        "num_pose_inliers": len(pose_inlier_matches),
        "num_triangulated": len(points_3d),
        "inlier_ratio": inlier_ratio,
        "score": score
    }

def select_initial_pair(
    pair_infos,
    all_results,
    min_pose_inliers=30,
    min_triangulated_points=30,
    min_angle_deg=3.0,
    max_angle_deg=70.0,
    focal_scale=1.2
):
    """
    从图像连接图 G 的所有边中选择最佳初始图像对。

    返回：
    best_result: 最佳初始图像对的完整信息。
    """

    candidate_results = []
    for pair, pair_info in pair_infos.items():
        result = evaluate_initial_pair(
            pair=pair,
            pair_info=pair_info,
            all_results=all_results,
            min_pose_inliers=min_pose_inliers,
            min_triangulated_points=min_triangulated_points,
            min_angle_deg=min_angle_deg,
            max_angle_deg=max_angle_deg,
            focal_scale=focal_scale
        )

        if result["success"]:
            candidate_results.append(result)
    if len(candidate_results) == 0:
        return None
    best_result = max(candidate_results, key=lambda x: x["score"])

    return best_result

def select_next_edge_for_pnp(
    pair_infos,
    registered_images,
    track_to_point3D,
    observation_to_track,
    min_common_points=20
):
    """
    从 G 中选择下一条用于 PnP 的边 e。

    选择标准：
        1. 这条边连接一张已注册图像和一张未注册图像；
        2. 这条边中的 tracks 有尽可能多已经被重建成 3D 点；
        3. 这些 3D 点可以和未注册图像中的 2D keypoints 构成 PnP 对应。

    返回：
        best_pair: 选中的边 (i, j)
        best_new_image: 需要注册的新图像 id
        best_registered_image: 已注册图像 id
        best_score: 可用 3D-2D 对应数量
    """

    best_pair = None
    best_new_image = None
    best_registered_image = None
    best_score = 0

    for pair, pair_info in pair_infos.items():
        i, j = pair
        i_registered = i in registered_images
        j_registered = j in registered_images

        # 必须是一张已注册，一张未注册
        if i_registered == j_registered:
            continue

        if i_registered:
            registered_image = i
            new_image = j
        else:
            registered_image = j
            new_image = i

        inlier_matches = pair_info["inlier_matches"]
        common_track_ids = set()
        for m in inlier_matches:
            # 注意 pair_infos 的 key 是 (i, j)，DMatch 的 queryIdx 属于 i，trainIdx 属于 j
            obs_i = (i, m.queryIdx)
            obs_j = (j, m.trainIdx)

            if new_image == i:
                obs_new = obs_i
            else:
                obs_new = obs_j

            track_id = observation_to_track.get(obs_new, None)

            if track_id is None:
                continue

            if track_id in track_to_point3D:
                common_track_ids.add(track_id)

        score = len(common_track_ids)
        if score > best_score:
            best_score = score
            best_pair = pair
            best_new_image = new_image
            best_registered_image = registered_image
    if best_score < min_common_points:
        return None, None, None, best_score

    return best_pair, best_new_image, best_registered_image, best_score

def collect_pnp_correspondences_from_edge(
    pair,
    pair_info,
    new_image_id,
    observation_to_track,
    track_to_point3D,
    points3D,
    all_results
):
    """
    从一条边中收集 PnP 所需的 3D-2D 对应关系。

    返回：
        object_points: N×3，已有三维点
        image_points: N×2，新图像中的二维点
        used_track_ids: 对应的 track_id
    """

    i, j = pair
    inlier_matches = pair_info["inlier_matches"]
    object_points = []
    image_points = []
    used_track_ids = []
    kp_new = all_results[new_image_id]["keypoints"]
    used = set()

    for m in inlier_matches:
        obs_i = (i, m.queryIdx)
        obs_j = (j, m.trainIdx)

        if new_image_id == i:
            obs_new = obs_i
            keypoint_id_new = m.queryIdx
        elif new_image_id == j:
            obs_new = obs_j
            keypoint_id_new = m.trainIdx
        else:
            continue

        track_id = observation_to_track.get(obs_new, None)

        if track_id is None:
            continue
        if track_id not in track_to_point3D:
            continue
        if track_id in used:
            continue

        point3D_id = track_to_point3D[track_id]
        X = points3D[point3D_id]["xyz"]
        x = kp_new[keypoint_id_new].pt

        object_points.append(X)
        image_points.append(x)
        used_track_ids.append(track_id)

        used.add(track_id)

    object_points = np.asarray(object_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)

    return object_points, image_points, used_track_ids

def register_image_by_pnp(
    pair,
    pair_info,
    new_image_id,
    observation_to_track,
    track_to_point3D,
    points3D,
    all_results,
    camera_intrinsics,
    min_pnp_points=20,
    reprojection_error=8.0,
    confidence=0.99,
    iterations_count=1000
):
    """
    使用 PnP RANSAC 估计新图像的相机位姿。

    返回：
        success
        R
        t
        pnp_inlier_track_ids
    """

    K = build_K_from_intrinsics(camera_intrinsics)

    object_points, image_points, used_track_ids = collect_pnp_correspondences_from_edge(
        pair=pair,
        pair_info=pair_info,
        new_image_id=new_image_id,
        observation_to_track=observation_to_track,
        track_to_point3D=track_to_point3D,
        points3D=points3D,
        all_results=all_results
    )

    if len(object_points) < min_pnp_points:
        return False, None, None, [], {
            "reason": "PnP 可用 3D-2D 对应点不足",
            "num_correspondences": len(object_points)
        }

    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=K,
            distCoeffs=None,
            iterationsCount=iterations_count,
            reprojectionError=reprojection_error,
            confidence=confidence,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
    except cv2.error as e:
        return False, None, None, [], {
            "reason": "solvePnPRansac 出错",
            "error": str(e)
        }

    if not success or inliers is None:
        return False, None, None, [], {
            "reason": "PnP 估计失败",
            "num_correspondences": len(object_points)
        }

    inliers = inliers.ravel()

    if len(inliers) < min_pnp_points:
        return False, None, None, [], {
            "reason": "PnP RANSAC 内点不足",
            "num_pnp_inliers": len(inliers)
        }

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)

    pnp_inlier_track_ids = [
        used_track_ids[idx] for idx in inliers
    ]

    return True, R, t, pnp_inlier_track_ids, {
        "num_correspondences": len(object_points),
        "num_pnp_inliers": len(inliers)
    }

def from_homogeneous(points_h, eps=1e-8):
    points_h = np.asarray(points_h, dtype=np.float64)
    w = points_h[:, -1]

    points = np.full(
        (points_h.shape[0], points_h.shape[1] - 1),
        np.nan,
        dtype=np.float64
    )

    valid = np.abs(w) > eps
    points[valid] = points_h[valid, :-1] / w[valid, None]

    return points

def project_point(K, R, t, X):
    X = X.reshape(3, 1)
    x = K @ (R @ X + t.reshape(3, 1))
    if abs(x[2, 0]) < 1e-12:
        return None
    return np.array([x[0, 0] / x[2, 0], x[1, 0] / x[2, 0]], dtype=np.float64)

def is_triangulated_point_valid(
    X,
    obs1,
    obs2,
    all_results,
    camera_poses,
    camera_intrinsics,
    reproj_error_threshold=5.0
):
    """
    检查三角化点是否有效。
    """

    if X is None:
        return False

    if not np.isfinite(X).all():
        return False

    K = build_K_from_intrinsics(camera_intrinsics)

    image_id1, kp_id1 = obs1
    image_id2, kp_id2 = obs2

    if image_id1 not in camera_poses:
        return False

    if image_id2 not in camera_poses:
        return False

    R1 = camera_poses[image_id1]["R"]
    t1 = camera_poses[image_id1]["t"]

    R2 = camera_poses[image_id2]["R"]
    t2 = camera_poses[image_id2]["t"]

    X_cam1 = R1 @ X.reshape(3, 1) + t1.reshape(3, 1)
    X_cam2 = R2 @ X.reshape(3, 1) + t2.reshape(3, 1)

    if X_cam1[2, 0] <= 0 or X_cam2[2, 0] <= 0:
        return False

    x1_proj = project_point(K, R1, t1, X)
    x2_proj = project_point(K, R2, t2, X)

    if x1_proj is None or x2_proj is None:
        return False

    kp1 = all_results[image_id1]["keypoints"][kp_id1].pt
    kp2 = all_results[image_id2]["keypoints"][kp_id2].pt

    x1_obs = np.array(kp1, dtype=np.float64)
    x2_obs = np.array(kp2, dtype=np.float64)

    err1 = np.linalg.norm(x1_proj - x1_obs)
    err2 = np.linalg.norm(x2_proj - x2_obs)

    if err1 > reproj_error_threshold or err2 > reproj_error_threshold:
        return False

    return True

def triangulate_two_observations(
    obs1,
    obs2,
    all_results,
    camera_poses,
    camera_intrinsics
):
    """
    根据两个已注册相机中的观测三角化一个三维点。

    obs1, obs2:
        (image_id, keypoint_id)
    """

    image_id1, kp_id1 = obs1
    image_id2, kp_id2 = obs2

    if image_id1 not in camera_poses:
        print(f"跳过三角化：image_id={image_id1} 没有相机位姿。")
        return None

    if image_id2 not in camera_poses:
        print(f"跳过三角化：image_id={image_id2} 没有相机位姿。")
        return None

    K = build_K_from_intrinsics(camera_intrinsics)

    R1 = camera_poses[image_id1]["R"]
    t1 = camera_poses[image_id1]["t"]

    R2 = camera_poses[image_id2]["R"]
    t2 = camera_poses[image_id2]["t"]

    P1 = build_projection_matrix(K, R1, t1)
    P2 = build_projection_matrix(K, R2, t2)

    x1 = np.array(
        all_results[image_id1]["keypoints"][kp_id1].pt,
        dtype=np.float64
    )

    x2 = np.array(
        all_results[image_id2]["keypoints"][kp_id2].pt,
        dtype=np.float64
    )

    points_4d = cv2.triangulatePoints(
        P1,
        P2,
        x1.reshape(2, 1),
        x2.reshape(2, 1)
    )

    X = from_homogeneous(points_4d.T)[0]

    return X

def triangulate_new_tracks_after_registering_image(
    new_image_id,
    pair_infos,
    valid_tracks,
    observation_to_track,
    track_to_point3D,
    point3D_to_track,
    points3D,
    camera_poses,
    registered_images,
    all_results,
    camera_intrinsics,
    reproj_error_threshold=5.0
):
    """
    新图像注册成功后，三角化新的 tracks。

    安全版本：
        1. 不仅检查 registered_images；
        2. 更重要的是检查 camera_poses 中是否真的有该图像的 R,t；
        3. 避免出现 KeyError。
    """

    new_points_count = 0

    # 如果新图像没有相机位姿，不能三角化
    if new_image_id not in camera_poses:
        print(f"错误：new_image_id={new_image_id} 不在 camera_poses 中，无法三角化。")
        return 0

    # 保证 registered_images 和 camera_poses 尽量一致
    registered_images = set(registered_images)
    pose_image_ids = set(camera_poses.keys())

    missing_pose_images = registered_images - pose_image_ids

    if len(missing_pose_images) > 0:
        print("警告：registered_images 中存在没有 camera_poses 的图像：")
        print(missing_pose_images)
        print("本次三角化将只使用 camera_poses 中已有位姿的图像。")

    next_point_id = 0
    if len(points3D) > 0:
        next_point_id = max(points3D.keys()) + 1

    for track_id, track in enumerate(valid_tracks):

        # 已经有 3D 点的 track 不再重复三角化
        if track_id in track_to_point3D:
            continue

        # track 中必须包含新注册图像的观测
        obs_new_list = [
            obs for obs in track
            if obs[0] == new_image_id
        ]

        if len(obs_new_list) == 0:
            continue

        obs_new = obs_new_list[0]

        # 找到该 track 中另一张已经有相机位姿的图像观测
        candidate_obs = [
            obs for obs in track
            if obs[0] in camera_poses and obs[0] != new_image_id
        ]

        if len(candidate_obs) == 0:
            continue

        # 简化处理：先选择第一个已有位姿的旧观测
        obs_old = candidate_obs[0]

        # 再次保险检查
        if obs_old[0] not in camera_poses:
            continue

        if obs_new[0] not in camera_poses:
            continue

        try:
            X = triangulate_two_observations(
                obs1=obs_old,
                obs2=obs_new,
                all_results=all_results,
                camera_poses=camera_poses,
                camera_intrinsics=camera_intrinsics
            )
        except KeyError as e:
            print(f"三角化跳过：缺少相机位姿 {e}")
            continue

        if X is None:
            continue

        valid = is_triangulated_point_valid(
            X=X,
            obs1=obs_old,
            obs2=obs_new,
            all_results=all_results,
            camera_poses=camera_poses,
            camera_intrinsics=camera_intrinsics,
            reproj_error_threshold=reproj_error_threshold
        )

        if not valid:
            continue

        point_id = next_point_id
        next_point_id += 1

        points3D[point_id] = {
            "xyz": X,
            "track_id": track_id,
            "observations": track
        }

        track_to_point3D[track_id] = point_id
        point3D_to_track[point_id] = track_id

        new_points_count += 1

    return new_points_count

def run_bundle_adjustment_fixed_K(
    camera_poses,
    points3D,
    registered_images,
    all_results,
    camera_intrinsics,
    fixed_image_id=None,
    max_nfev=50
):
    """
    简化版 Bundle Adjustment。

    优化变量：
        1. 已注册相机的 rvec, tvec
        2. 已重建三维点 xyz

    固定：
        1. 相机内参 K
        2. fixed_image_id 对应的相机位姿，用于固定坐标系尺度和规范自由度

    注意：
        这是教学版 BA，适合你当前程序接入。
        后续可以扩展为优化 K 的 BA。
    """

    if fixed_image_id is None:
        fixed_image_id = min(registered_images)

    K = build_K_from_intrinsics(camera_intrinsics)

    # 只优化有足够观测的点
    point_ids = list(points3D.keys())
    image_ids = sorted(list(registered_images))

    variable_image_ids = [
        image_id for image_id in image_ids
        if image_id != fixed_image_id
    ]

    image_id_to_var_idx = {
        image_id: idx for idx, image_id in enumerate(variable_image_ids)
    }

    point_id_to_var_idx = {
        point_id: idx for idx, point_id in enumerate(point_ids)
    }

    # 收集 BA 观测
    observations = []

    for point_id in point_ids:
        point = points3D[point_id]
        for obs in point["observations"]:
            image_id, kp_id = obs

            if image_id not in registered_images:
                continue

            x_obs = np.array(
                all_results[image_id]["keypoints"][kp_id].pt,
                dtype=np.float64
            )

            observations.append((image_id, point_id, x_obs))

    if len(observations) < 10:
        print("BA 观测数量太少，跳过。")
        return camera_poses, points3D

    # 打包优化变量
    x0_list = []

    for image_id in variable_image_ids:
        R = camera_poses[image_id]["R"]
        t = camera_poses[image_id]["t"].reshape(3, 1)

        rvec, _ = cv2.Rodrigues(R)

        x0_list.extend(rvec.ravel())
        x0_list.extend(t.ravel())

    for point_id in point_ids:
        X = points3D[point_id]["xyz"]
        x0_list.extend(X.ravel())

    x0 = np.array(x0_list, dtype=np.float64)

    num_cameras = len(variable_image_ids)
    num_points = len(point_ids)

    def unpack_params(params):
        poses = {}

        offset = 0

        # 固定相机
        poses[fixed_image_id] = {
            "R": camera_poses[fixed_image_id]["R"],
            "t": camera_poses[fixed_image_id]["t"]
        }

        # 可变相机
        for image_id in variable_image_ids:
            rvec = params[offset:offset + 3]
            offset += 3

            tvec = params[offset:offset + 3].reshape(3, 1)
            offset += 3

            R, _ = cv2.Rodrigues(rvec)

            poses[image_id] = {
                "R": R,
                "t": tvec
            }

        # 三维点
        points = {}

        for point_id in point_ids:
            X = params[offset:offset + 3]
            offset += 3
            points[point_id] = X

        return poses, points

    def residual_function(params):
        poses, points = unpack_params(params)

        residuals = []

        for image_id, point_id, x_obs in observations:
            R = poses[image_id]["R"]
            t = poses[image_id]["t"]
            X = points[point_id]

            x_proj = project_point(K, R, t, X)

            if x_proj is None:
                residuals.extend([1000.0, 1000.0])
                continue

            residual = x_proj - x_obs

            residuals.extend(residual.tolist())

        return np.array(residuals, dtype=np.float64)

    result = least_squares(
        residual_function,
        x0,
        loss="huber",
        f_scale=3.0,
        max_nfev=max_nfev,
        verbose=0
    )

    optimized_poses, optimized_points = unpack_params(result.x)

    # 写回 camera_poses
    for image_id in optimized_poses:
        camera_poses[image_id]["R"] = optimized_poses[image_id]["R"]
        camera_poses[image_id]["t"] = optimized_poses[image_id]["t"]

    # 写回 points3D
    for point_id in optimized_points:
        points3D[point_id]["xyz"] = optimized_points[point_id]

    print(
        f"BA 完成：cost={result.cost:.4f}, "
        f"observations={len(observations)}, "
        f"cameras={len(registered_images)}, "
        f"points={len(points3D)}"
    )

    return camera_poses, points3D

def project_point_with_f(f, cx, cy, R, t, X):
    """
    使用 SIMPLE_PINHOLE 模型投影一个三维点。

    K = [ f  0  cx
          0  f  cy
          0  0   1 ]

    返回：
        图像坐标 [u, v]
    """

    X = X.reshape(3, 1)
    t = t.reshape(3, 1)

    X_cam = R @ X + t

    z = X_cam[2, 0]

    if abs(z) < 1e-12:
        return None

    x = X_cam[0, 0] / z
    y = X_cam[1, 0] / z

    u = f * x + cx
    v = f * y + cy

    return np.array([u, v], dtype=np.float64)

def run_bundle_adjustment_refine_focal(
    camera_poses,
    points3D,
    registered_images,
    all_results,
    camera_intrinsics,
    fixed_image_id=None,
    max_nfev=50
):
    """
    Bundle Adjustment：优化焦距 f、相机位姿 R,t、三维点 X。

    优化变量：
        1. 焦距 f
        2. 除 fixed_image_id 之外的相机位姿 rvec, tvec
        3. 所有三维点 xyz

    固定变量：
        1. 主点 cx, cy
        2. fixed_image_id 的相机位姿

    当前相机模型：
        SIMPLE_PINHOLE，即 fx = fy = f
    """

    if fixed_image_id is None:
        fixed_image_id = min(registered_images)

    image_ids = sorted(list(registered_images))
    point_ids = list(points3D.keys())

    if len(image_ids) < 5:
        print("注册图像数量较少，不优化焦距，改用固定 K BA。")
        camera_poses, points3D = run_bundle_adjustment_fixed_K(
            camera_poses=camera_poses,
            points3D=points3D,
            registered_images=registered_images,
            all_results=all_results,
            camera_intrinsics=camera_intrinsics,
            fixed_image_id=fixed_image_id,
            max_nfev=max_nfev
        )
        return camera_poses, points3D, camera_intrinsics

    if len(point_ids) < 100:
        print("三维点数量较少，不优化焦距，改用固定 K BA。")
        camera_poses, points3D = run_bundle_adjustment_fixed_K(
            camera_poses=camera_poses,
            points3D=points3D,
            registered_images=registered_images,
            all_results=all_results,
            camera_intrinsics=camera_intrinsics,
            fixed_image_id=fixed_image_id,
            max_nfev=max_nfev
        )
        return camera_poses, points3D, camera_intrinsics

    variable_image_ids = [
        image_id for image_id in image_ids
        if image_id != fixed_image_id
    ]

    f_init = float(camera_intrinsics["f"])
    cx = float(camera_intrinsics["cx"])
    cy = float(camera_intrinsics["cy"])

    width = camera_intrinsics.get("width", None)
    height = camera_intrinsics.get("height", None)

    if width is not None and height is not None:
        max_dim = max(width, height)
        min_f = 0.3 * max_dim
        max_f = 5.0 * max_dim
    else:
        min_f = 0.2 * f_init
        max_f = 5.0 * f_init

    # -------------------------------
    # 1. 收集 BA 观测
    # -------------------------------
    observations = []

    for point_id in point_ids:
        point = points3D[point_id]

        for obs in point["observations"]:
            image_id, kp_id = obs

            if image_id not in registered_images:
                continue

            x_obs = np.array(
                all_results[image_id]["keypoints"][kp_id].pt,
                dtype=np.float64
            )

            observations.append((image_id, point_id, x_obs))

    if len(observations) < 300:
        print("BA 观测数量较少，不优化焦距，改用固定 K BA。")
        camera_poses, points3D = run_bundle_adjustment_fixed_K(
            camera_poses=camera_poses,
            points3D=points3D,
            registered_images=registered_images,
            all_results=all_results,
            camera_intrinsics=camera_intrinsics,
            fixed_image_id=fixed_image_id,
            max_nfev=max_nfev
        )
        return camera_poses, points3D, camera_intrinsics

    # -------------------------------
    # 2. 打包优化变量
    # -------------------------------
    x0_list = []

    # 第一个变量：焦距 f
    x0_list.append(f_init)

    # 相机位姿变量
    for image_id in variable_image_ids:
        R = camera_poses[image_id]["R"]
        t = camera_poses[image_id]["t"].reshape(3, 1)

        rvec, _ = cv2.Rodrigues(R)

        x0_list.extend(rvec.ravel())
        x0_list.extend(t.ravel())

    # 三维点变量
    for point_id in point_ids:
        X = points3D[point_id]["xyz"]
        x0_list.extend(X.ravel())

    x0 = np.array(x0_list, dtype=np.float64)

    # -------------------------------
    # 3. 设置上下界
    # -------------------------------
    lower_bounds = []
    upper_bounds = []

    # 焦距范围
    lower_bounds.append(min_f)
    upper_bounds.append(max_f)

    # 相机位姿不限制
    for _ in variable_image_ids:
        lower_bounds.extend([-np.inf] * 6)
        upper_bounds.extend([np.inf] * 6)

    # 三维点不限制
    for _ in point_ids:
        lower_bounds.extend([-np.inf] * 3)
        upper_bounds.extend([np.inf] * 3)

    lower_bounds = np.array(lower_bounds, dtype=np.float64)
    upper_bounds = np.array(upper_bounds, dtype=np.float64)

    # -------------------------------
    # 4. 解包函数
    # -------------------------------
    def unpack_params(params):
        offset = 0

        f = params[offset]
        offset += 1

        poses = {}

        poses[fixed_image_id] = {
            "R": camera_poses[fixed_image_id]["R"],
            "t": camera_poses[fixed_image_id]["t"]
        }

        for image_id in variable_image_ids:
            rvec = params[offset:offset + 3]
            offset += 3

            tvec = params[offset:offset + 3].reshape(3, 1)
            offset += 3

            R, _ = cv2.Rodrigues(rvec)

            poses[image_id] = {
                "R": R,
                "t": tvec
            }

        points = {}

        for point_id in point_ids:
            X = params[offset:offset + 3]
            offset += 3

            points[point_id] = X

        return f, poses, points

    # -------------------------------
    # 5. 残差函数
    # -------------------------------
    def residual_function(params):
        f, poses, points = unpack_params(params)

        residuals = []

        for image_id, point_id, x_obs in observations:
            R = poses[image_id]["R"]
            t = poses[image_id]["t"]
            X = points[point_id]

            x_proj = project_point_with_f(
                f=f,
                cx=cx,
                cy=cy,
                R=R,
                t=t,
                X=X
            )

            if x_proj is None:
                residuals.extend([1000.0, 1000.0])
                continue

            residual = x_proj - x_obs
            residuals.extend(residual.tolist())

        return np.array(residuals, dtype=np.float64)

    # -------------------------------
    # 6. 执行优化
    # -------------------------------
    result = least_squares(
        residual_function,
        x0,
        bounds=(lower_bounds, upper_bounds),
        loss="huber",
        f_scale=3.0,
        max_nfev=max_nfev,
        verbose=0
    )

    f_opt, optimized_poses, optimized_points = unpack_params(result.x)

    # -------------------------------
    # 7. 写回结果
    # -------------------------------
    camera_intrinsics["f"] = float(f_opt)

    for image_id in optimized_poses:
        camera_poses[image_id]["R"] = optimized_poses[image_id]["R"]
        camera_poses[image_id]["t"] = optimized_poses[image_id]["t"]

    for point_id in optimized_points:
        points3D[point_id]["xyz"] = optimized_points[point_id]

    print(
        f"BA refine focal 完成："
        f"cost={result.cost:.4f}, "
        f"f: {f_init:.2f} -> {f_opt:.2f}, "
        f"observations={len(observations)}, "
        f"cameras={len(registered_images)}, "
        f"points={len(points3D)}"
    )

    return camera_poses, points3D, camera_intrinsics

def run_sfm_bundle_adjustment(
    camera_poses,
    points3D,
    registered_images,
    all_results,
    camera_intrinsics,
    fixed_image_id=None,
    refine_focal_min_images=5,
    refine_focal_min_points=150,
    refine_focal_min_observations=500,
    fixed_K_max_nfev=30,
    refine_focal_max_nfev=50
):
    """
    SfM 中的 BA 调度函数。

    根据当前重建状态自动选择：
        1. 固定 K 的 BA
        2. 优化焦距 f 的 BA

    返回：
        camera_poses
        points3D
        camera_intrinsics
    """

    if fixed_image_id is None:
        fixed_image_id = min(registered_images)

    # 统计当前观测数量
    num_observations = 0

    for point_id, point in points3D.items():
        for obs in point["observations"]:
            image_id, kp_id = obs

            if image_id in registered_images:
                num_observations += 1

    num_registered_images = len(registered_images)
    num_points = len(points3D)

    print(
        f"BA 调度："
        f"images={num_registered_images}, "
        f"points={num_points}, "
        f"observations={num_observations}, "
        f"current_f={camera_intrinsics['f']:.3f}"
    )

    can_refine_focal = (
        num_registered_images >= refine_focal_min_images
        and num_points >= refine_focal_min_points
        and num_observations >= refine_focal_min_observations
        and camera_intrinsics.get("refine_focal_length", True)
    )

    if can_refine_focal:
        print("执行 BA：优化焦距 f + 相机位姿 R,t + 三维点 X")

        camera_poses, points3D, camera_intrinsics = run_bundle_adjustment_refine_focal(
            camera_poses=camera_poses,
            points3D=points3D,
            registered_images=registered_images,
            all_results=all_results,
            camera_intrinsics=camera_intrinsics,
            fixed_image_id=fixed_image_id,
            max_nfev=refine_focal_max_nfev
        )

    else:
        print("执行 BA：固定 K，只优化相机位姿 R,t + 三维点 X")

        camera_poses, points3D = run_bundle_adjustment_fixed_K(
            camera_poses=camera_poses,
            points3D=points3D,
            registered_images=registered_images,
            all_results=all_results,
            camera_intrinsics=camera_intrinsics,
            fixed_image_id=fixed_image_id,
            max_nfev=fixed_K_max_nfev
        )

    return camera_poses, points3D, camera_intrinsics

def incremental_sfm_expansion(
    pair_infos,
    valid_tracks,
    observation_to_track,
    camera_poses,
    registered_images,
    points3D,
    track_to_point3D,
    point3D_to_track,
    all_results,
    camera_intrinsics,
    min_common_points=20,
    min_pnp_points=20,
    ba_every_iteration=True,
    global_ba_every=5
):
    """
    增量式 SfM 扩展。

    按照你的流程：

        while G 中还有边：
            1. 从 G 中选取边 e，使 track(e) 与已重建 3D 点交集最大
            2. 用 PnP 方法估计新图像相机位姿
            3. 三角化新的 tracks
            4. 删除 G 中的边 e
            5. 执行 Bundle Adjustment

    BA 策略：
        前期固定 K，只优化 R,t,X；
        后期优化焦距 f，同时优化 R,t,X。
    """

    remaining_edges = dict(pair_infos)

    iteration = 0

    while len(remaining_edges) > 0:
        registered_images = set(camera_poses.keys())
        iteration += 1

        print(f"\n========== SfM 增量迭代 {iteration} ==========")
        print(f"剩余边数量：{len(remaining_edges)}")
        print(f"已注册图像数量：{len(registered_images)}")
        print(f"当前三维点数量：{len(points3D)}")
        print(f"当前焦距 f：{camera_intrinsics['f']:.3f}")

        # ------------------------------------------------
        # 1. 从 G 中选取边 e
        # ------------------------------------------------
        pair, new_image_id, registered_image_id, score = select_next_edge_for_pnp(
            pair_infos=remaining_edges,
            registered_images=registered_images,
            track_to_point3D=track_to_point3D,
            observation_to_track=observation_to_track,
            min_common_points=min_common_points
        )

        if pair is None:
            print("没有更多满足 PnP 条件的边，增量 SfM 结束。")
            break

        print(
            f"选择边 e={pair}, "
            f"已注册图像={registered_image_id}, "
            f"新图像={new_image_id}, "
            f"可用 3D-2D 数量={score}"
        )

        pair_info = remaining_edges[pair]

        # ------------------------------------------------
        # 2. 用 PnP 方法估计新图像位姿
        # ------------------------------------------------
        success, R, t, pnp_inlier_track_ids, pnp_info = register_image_by_pnp(
            pair=pair,
            pair_info=pair_info,
            new_image_id=new_image_id,
            observation_to_track=observation_to_track,
            track_to_point3D=track_to_point3D,
            points3D=points3D,
            all_results=all_results,
            camera_intrinsics=camera_intrinsics,
            min_pnp_points=min_pnp_points
        )

        if not success:
            print(f"PnP 注册失败：{pnp_info}")

            # 按你的流程，失败边也从 G 中删除
            del remaining_edges[pair]
            print(f"删除失败边 e={pair}")

            continue

        camera_poses[new_image_id] = {
            "R": R,
            "t": t
        }

        registered_images.add(new_image_id)

        print(
            f"PnP 注册成功：image={new_image_id}, "
            f"PnP inliers={pnp_info['num_pnp_inliers']}"
        )

        # ------------------------------------------------
        # 3. 三角化新的 tracks
        # ------------------------------------------------
        new_points_count = triangulate_new_tracks_after_registering_image(
            new_image_id=new_image_id,
            pair_infos=remaining_edges,
            valid_tracks=valid_tracks,
            observation_to_track=observation_to_track,
            track_to_point3D=track_to_point3D,
            point3D_to_track=point3D_to_track,
            points3D=points3D,
            camera_poses=camera_poses,
            registered_images=registered_images,
            all_results=all_results,
            camera_intrinsics=camera_intrinsics,
            reproj_error_threshold=5.0
        )

        print(f"新增三角化点数量：{new_points_count}")

        # ------------------------------------------------
        # 4. 删除 G 中的边 e
        # ------------------------------------------------
        del remaining_edges[pair]
        print(f"删除已处理边 e={pair}")

        # ------------------------------------------------
        # 5. 每轮 BA
        # ------------------------------------------------
        if ba_every_iteration:
            camera_poses, points3D, camera_intrinsics = run_sfm_bundle_adjustment(
                camera_poses=camera_poses,
                points3D=points3D,
                registered_images=registered_images,
                all_results=all_results,
                camera_intrinsics=camera_intrinsics,
                fixed_image_id=min(registered_images),
                refine_focal_min_images=5,
                refine_focal_min_points=150,
                refine_focal_min_observations=500,
                fixed_K_max_nfev=3,
                refine_focal_max_nfev=3
            )

            print(f"每轮 BA 后焦距 f：{camera_intrinsics['f']:.3f}")

        # ------------------------------------------------
        # 6. 周期性全局 BA
        # ------------------------------------------------
        if global_ba_every is not None and iteration % global_ba_every == 0:
            print("执行周期性全局 BA")

            camera_poses, points3D, camera_intrinsics = run_sfm_bundle_adjustment(
                camera_poses=camera_poses,
                points3D=points3D,
                registered_images=registered_images,
                all_results=all_results,
                camera_intrinsics=camera_intrinsics,
                fixed_image_id=min(registered_images),
                refine_focal_min_images=5,
                refine_focal_min_points=150,
                refine_focal_min_observations=500,
                fixed_K_max_nfev=3,
                refine_focal_max_nfev=3
            )

            print(f"周期性 BA 后焦距 f：{camera_intrinsics['f']:.3f}")

    # ------------------------------------------------
    # 7. 最终全局 BA
    # ------------------------------------------------
    if len(registered_images) >= 2 and len(points3D) >= 10:
        print("\n执行最终全局 BA")

        camera_poses, points3D, camera_intrinsics = run_sfm_bundle_adjustment(
            camera_poses=camera_poses,
            points3D=points3D,
            registered_images=registered_images,
            all_results=all_results,
            camera_intrinsics=camera_intrinsics,
            fixed_image_id=min(registered_images),
            refine_focal_min_images=5,
            refine_focal_min_points=150,
            refine_focal_min_observations=500,
            fixed_K_max_nfev=3,
            refine_focal_max_nfev=3
        )

    print("\n========== 增量 SfM 结束 ==========")
    print(f"最终注册图像数量：{len(registered_images)}")
    print(f"最终三维点数量：{len(points3D)}")
    print(f"最终焦距 f：{camera_intrinsics['f']:.3f}")

    return (
        camera_poses,
        registered_images,
        points3D,
        track_to_point3D,
        point3D_to_track,
        camera_intrinsics
    )

def get_point_color_from_observations(point_info, all_results):
    """
    根据三维点的 observations 从图像中取颜色。

    使用该点第一个有效观测位置的像素颜色。
    OpenCV 图像格式是 BGR，这里转换为 RGB。
    """

    observations = point_info.get("observations", [])

    for image_id, kp_id in observations:
        if image_id < 0 or image_id >= len(all_results):
            continue

        image = all_results[image_id]["image"]
        keypoints = all_results[image_id]["keypoints"]

        if kp_id < 0 or kp_id >= len(keypoints):
            continue

        h, w = image.shape[:2]

        x, y = keypoints[kp_id].pt
        x = int(round(x))
        y = int(round(y))

        if 0 <= x < w and 0 <= y < h:
            b, g, r = image[y, x]
            return int(r), int(g), int(b)

    return 255, 255, 255

def save_colored_points3D_to_ply(points3D, all_results, save_path):
    """
    将 points3D 保存为带颜色的 PLY 点云。
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    valid_points = []

    for point_id, point_info in points3D.items():
        X = np.asarray(point_info["xyz"], dtype=np.float64).reshape(3)

        if not np.isfinite(X).all():
            continue

        r, g, b = get_point_color_from_observations(
            point_info=point_info,
            all_results=all_results
        )

        valid_points.append((X[0], X[1], X[2], r, g, b))

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(valid_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for x, y, z, r, g, b in valid_points:
            f.write(
                f"{x:.6f} {y:.6f} {z:.6f} "
                f"{int(r)} {int(g)} {int(b)}\n"
            )

    print(f"彩色三维点云已保存：{save_path}")
    print(f"有效三维点数量：{len(valid_points)}")

def main():
    # 先设置输入输出路径，并创建相应的文件夹
    image_folder = "image"
    output_folder = "output"
    create_folder(output_folder)

    # 获取所有图片路径
    image_files = get_image_files(image_folder)
    if len(image_files) == 0:
        print("没有在 image 文件夹中找到图片。")
        return
    print(f"共找到 {len(image_files)} 张图片。")

    # 创建 SIFT 对象
    sift = cv2.SIFT_create()

    # 用于保存所有图片的特征结果
    all_results = []

    # 提取 SIFT 特征
    for index, image_path in enumerate(image_files):
        print(f"[{index + 1}/{len(image_files)}] 正在处理：{image_path.name}")          #提示处理进度, 便于查看报错
        image, gray, keypoints, descriptors = extract_sift_features(image_path, sift)   #提取 SIFT 特征，得到原图、灰度图、关键点和描述子

        if image is None:
            print("没有检测到关键点")
            continue

        print(f"检测到关键点数量：{len(keypoints)}")

        # 保存到总结果列表中，后续用于匹配
        all_results.append({
            "image_path": image_path,
            "image": image,
            "gray": gray,
            "keypoints": keypoints,
            "descriptors": descriptors
        })

    # 相邻图片之间做特征匹配
    print("\n开始进行相邻图片的 SIFT 匹配...")
    min_good_matches = 50  # 小于这个数量，说明两张图初步匹配关系较弱，不对其中的匹配点进行后续处理
    min_inlier_matches = 30 # 小于这个数量，说明两张图的几何关系较弱，不将其中的匹配点加入 Tracks 构建
    min_inlier_ratio = 0.25 # 内点比例小于这个值，说明两张图的几何关系较弱，不将其中的匹配点加入 Tracks 构建
    pair_infos = {}
    track_builder = UnionFind()    

    for i in range(len(all_results)):
        for j in range(i + 1, len(all_results)):
            result1 = all_results[i]
            result2 = all_results[j]

            img1 = result1["image"]
            img2 = result2["image"]

            kp1 = result1["keypoints"]
            kp2 = result2["keypoints"]

            desc1 = result1["descriptors"]
            desc2 = result2["descriptors"]

            name1 = result1["image_path"].stem
            name2 = result2["image_path"].stem

            # 第一步：SIFT 描述子匹配
            prime_matches = match_two_images(desc1, desc2)
            num_prime_matches = len(prime_matches)
                
            print(f"\n{name1} 与 {name2}")
            print(f"SIFT ratio test 后匹配点数量：{(num_prime_matches)}")
                
            if num_prime_matches < min_good_matches: #如果初步匹配点数量过少，说明两张图的内容差异较大，跳过后续步骤
                continue

            # 第二步：进行F矩阵的估计
            F, inlier_matches, inlier_mask, info = estimate_fundamental_matrix(kp1, kp2, prime_matches)

            if F is None or inlier_matches is None:
                continue
            num_inliers = len(inlier_matches)
            inlier_ratio = info["inlier_ratio"]
            if num_inliers < min_inlier_matches or inlier_ratio < min_inlier_ratio:
                continue

            # 第三步：将内点匹配加入 Tracks 构建
            pair_infos[(i, j)] = {
                "image1": i,
                "image2": j,
                "F": F,
                "prime_matches": prime_matches,
                "inlier_matches": inlier_matches,
                "inlier_mask": inlier_mask,
                "num_prime_matches": num_prime_matches,
                "num_inliers": num_inliers,
                "inlier_ratio": inlier_ratio,
                "weight": info["best_score"] * inlier_ratio,
                "info": info
            }

            for m in inlier_matches:
                obs1 = (i, m.queryIdx)
                obs2 = (j, m.trainIdx)
                track_builder.union(obs1, obs2)

    tracks = build_tracks_from_union_find(track_builder)
    print(f"初始 tracks 数量：{len(tracks)}")
    valid_tracks = filter_valid_tracks(tracks, min_track_length=3) # 过滤掉长度小于 3 的 track，以及在同一张图像中有多个观测点的 track
    print(f"过滤后有效的 tracks 数量：{len(valid_tracks)}")
    observation_to_track = build_observation_to_track(valid_tracks) # 建立 observation 到 track_id 的映射

    # 从图像连接图 G 的所有边中选择最佳初始图像对
    initial_pair_result = select_initial_pair(
        pair_infos=pair_infos,
        all_results=all_results,
        min_pose_inliers=30,
        min_triangulated_points=30,
        min_angle_deg=3.0,
        max_angle_deg=70.0,
        focal_scale=1.2
    )
    if initial_pair_result is None:
        print("没有找到满足视差角约束的初始图像对。")
        return
    print("\n成功选择初始图像对：")
    print("pair:", initial_pair_result["pair"])

    # 初始化全局 SfM 状态
    init_i, init_j = initial_pair_result["pair"]
    K_init = initial_pair_result["K"]
    R_init = initial_pair_result["R"]
    t_init = initial_pair_result["t"].reshape(3, 1)
    camera_intrinsics = {
        "K": K_init,
        "fx": K_init[0, 0],
        "fy": K_init[1, 1],
        "cx": K_init[0, 2],
        "cy": K_init[1, 2],
        "refine_in_BA": True
    }
    camera_poses = {}
    camera_poses[init_i] = {
        "R": np.eye(3, dtype=np.float64),
        "t": np.zeros((3, 1), dtype=np.float64)
    }
    registered_images = set([init_i, init_j]) # 已经注册到 SfM 中的图像编号集合
    initial_points_3d = initial_pair_result["points_3d"]
    initial_matches = initial_pair_result["triangulated_matches"] 

    print("\n初始化相机位姿完成：")
    print(f"已注册图像：{registered_images}")
    print("初始 K：")
    print(K_init)

    # 10. 建立初始 track_id -> point3D_id 映射
    points3D = {}
    track_to_point3D = {}
    point3D_to_track = {}
    point_id = 0

    for X, m in zip(initial_points_3d, initial_matches):
        obs1 = (init_i, m.queryIdx)
        obs2 = (init_j, m.trainIdx)
        track_id_1 = observation_to_track.get(obs1, None)
        track_id_2 = observation_to_track.get(obs2, None)

        if track_id_1 is None or track_id_2 is None:
            continue
        if track_id_1 != track_id_2:
            continue
        track_id = track_id_1
        if track_id in track_to_point3D:
            continue

        points3D[point_id] = {
            "xyz": X,
            "track_id": track_id,
            "observations": valid_tracks[track_id]
        }
        track_to_point3D[track_id] = point_id
        point3D_to_track[point_id] = track_id
        point_id += 1

    print(f"初始三维点数量：{len(points3D)}")

    camera_intrinsics = {
        "model": "SIMPLE_PINHOLE",
        "width": all_results[init_i]["image"].shape[1],
        "height": all_results[init_i]["image"].shape[0],
        "f": float(K_init[0, 0]),
        "cx": float(K_init[0, 2]),
        "cy": float(K_init[1, 2]),
        "refine_focal_length": True,
        "refine_principal_point": False,
        "refine_extra_params": False
    }

    (
        camera_poses,
        registered_images,
        points3D,
        track_to_point3D,
        point3D_to_track,
        camera_intrinsics
    ) = incremental_sfm_expansion(
        pair_infos=pair_infos,
        valid_tracks=valid_tracks,
        observation_to_track=observation_to_track,
        camera_poses=camera_poses,
        registered_images=registered_images,
        points3D=points3D,
        track_to_point3D=track_to_point3D,
        point3D_to_track=point3D_to_track,
        all_results=all_results,
        camera_intrinsics=camera_intrinsics,
        min_common_points=20,
        min_pnp_points=20,
        ba_every_iteration=True,
        global_ba_every=5
    )

    save_colored_points3D_to_ply(
        points3D=points3D,
        all_results=all_results,
        save_path=Path(output_folder) / "sparse_points_colored.ply"
    )

    print("\n全部处理完成。")

if __name__ == "__main__":
    main()