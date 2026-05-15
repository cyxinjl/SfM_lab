import cv2
import numpy as np
import math
from pathlib import Path
import random
from collections import defaultdict
import networkx as nx

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

    print("\n全部处理完成。")

if __name__ == "__main__":
    main()