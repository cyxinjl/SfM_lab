import cv2
import numpy as np
import os
from pathlib import Path

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


def save_keypoint_image(image, keypoints, save_path):   #提取出保有关键点的图像，并保存到指定路径
    keypoint_image = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )                                                   #使用 opencv 函数将关键点绘制在原图上

    cv2.imwrite(str(save_path), keypoint_image)         #将绘制了关键点的图像保存到指定路径


def match_two_images(desc1, desc2):         #对两张图的SIFT描述子进行匹配的函数，输入两张图的描述子，输出匹配结果列表
    if desc1 is None or desc2 is None:
        return []
    if len(desc1) < 2 or len(desc2) < 2:
        return []

    bf = cv2.BFMatcher(cv2.NORM_L2)         #创建匹配器对象，并用欧式距离衡量
    # knnMatch 中 k=2，表示对每个描述子找两个最相近的候选匹配
    matches = bf.knnMatch(desc1, desc2, k=2) #找出最优和次优匹配
    good_matches = []
    ratio_threshold = 0.75 # SIFT 匹配的距离比率阈值，越小表示匹配越严格
    # 如果最近邻明显优于次近邻，认为匹配可靠
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    return good_matches


def save_match_image(img1, kp1, img2, kp2, good_matches, save_path, max_draw_matches=50):
    #绘制两张图像的特征匹配图
    good_matches = sorted(good_matches, key=lambda x: x.distance) #按照匹配距离排序，距离越小表示匹配越好
    matches_to_draw = good_matches[:max_draw_matches]

    match_image = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imwrite(str(save_path), match_image)

def estimate_fundamental_matrix(kp1, kp2, good_matches, ransac_threshold=3.0, confidence=0.99):
    """
    根据两张图片的 SIFT 匹配点估计基础矩阵 F，并用 RANSAC 筛选几何内点。

    参数：
    kp1: 第一张图的关键点列表
    kp2: 第二张图的关键点列表
    good_matches: 初步 SIFT 得到的匹配列表
    ransac_threshold: RANSAC 重投影误差阈值，单位是像素
    confidence: RANSAC 置信度

    返回：
    F: 基础矩阵，通常是 3×3
    inlier_matches: 通过基础矩阵几何验证的匹配点
    mask: 每个匹配是否为内点的标记
    """

    # 基础矩阵估计至少需要 8 对匹配点
    if len(good_matches) < 8:
        return None, [], None

    # 取出特征点的坐标
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]) 
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    try:
        F, mask = cv2.findFundamentalMat(
            pts1,
            pts2,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=ransac_threshold,
            confidence=confidence
        )  #使用 RANSAC 方法估计基础矩阵 F，并返回每个匹配点是否为内点的 mask。mask 是一个与 good_matches 长度相同的数组，值为 1 表示对应匹配是内点，0 表示外点。
    except cv2.error as e:      #部分情况会出现未知原因的报错，直接跳过
        print("    cv2.findFundamentalMat 出错，跳过这组图片。")
        print("    错误信息：", e)
        return None, [], None


    if F is None or mask is None:
        return None, [], None

    mask = mask.ravel()

    inlier_matches = [
        m for m, keep in zip(good_matches, mask) if keep == 1
    ]

    return F, inlier_matches, mask

def create_approx_camera_matrix(image):  #通过图像大小近似相机内参矩阵K构造
    h, w = image.shape[:2]      #读取图像大小
    fx = 1.2 * max(w, h)
    fy = 1.2 * max(w, h)
    cx = w / 2.0
    cy = h / 2.0
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ], dtype=np.float64)

    return K

def rotation_matrix_to_euler_angles(R):  #  将旋转矩阵转换为欧拉角，单位为度。[rx, ry, rz],分别近似表示绕 x、y、z 轴的旋转角度。
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6  #如果 sy 非常小，说明旋转矩阵接近于奇异状态，此时无法通过常规方法计算欧拉角，需要特殊处理
    if not singular:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0

    return np.degrees(np.array([rx, ry, rz]))

def recover_camera_pose(kp1, kp2, matches, K, ransac_threshold=1.0, confidence=0.999): # 根据两张图片的匹配点和相机内参 K，恢复第二张相机相对于第一张相机的姿态。
    """
    参数：
    kp1: 第一张图的关键点列表
    kp2: 第二张图的关键点列表
    matches: 传入基础矩阵 RANSAC 筛选后的 inlier_matches
    K: 相机内参矩阵
    ransac_threshold: 本质矩阵 RANSAC 阈值
    confidence: RANSAC 置信度

    返回：
    E: 本质矩阵
    R: 第二张相机相对于第一张相机的旋转矩阵
    t: 第二张相机相对于第一张相机的平移方向
    pose_inlier_matches: recoverPose 认为有效的匹配点
    pose_mask: 姿态恢复阶段的内点标记
    """

    if matches is None or len(matches) < 5:     # 本质矩阵需要五对点才能够
        return None, None, None, [], None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    if pts1.shape[0] < 5 or pts2.shape[0] < 5:
        return None, None, None, [], None

    if pts1.shape != pts2.shape:
        return None, None, None, [], None

    if not np.isfinite(pts1).all() or not np.isfinite(pts2).all():
        return None, None, None, [], None

    try:
        E, mask_E = cv2.findEssentialMat(
            pts1,
            pts2,
            cameraMatrix=K,
            method=cv2.RANSAC,
            prob=confidence,
            threshold=ransac_threshold
        )       # 先轨迹本质矩阵E，并得到每个匹配点是否为本质矩阵内点的 mask_E。mask_E 是一个与 matches 长度相同的数组，值为 1 表示对应匹配是本质矩阵内点，0 表示外点。
    except cv2.error as e:
        print("    cv2.findEssentialMat 出错，跳过姿态恢复。")
        print("    错误信息：", e)
        return None, None, None, [], None

    if E is None or mask_E is None:
        return None, None, None, [], None

    # 有时 OpenCV 会返回多个候选 E，例如形状为 6×3、9×3，这里取第一个 3×3 本质矩阵
    if E.shape != (3, 3):
        E = E[:3, :3]

    try:
        retval, R, t, pose_mask = cv2.recoverPose(
            E,
            pts1,
            pts2,
            cameraMatrix=K,
            mask=mask_E
        )       # 使用 recoverPose 从本质矩阵 E 中恢复相机姿态 R 和 t，并得到每个匹配点是否为姿态恢复内点的 pose_mask。pose_mask 是一个与 matches 长度相同的数组，值为 1 表示对应匹配是姿态恢复内点，0 表示外点。
    except cv2.error as e:
        print("    cv2.recoverPose 出错，跳过姿态恢复。")
        print("    错误信息：", e)
        return None, None, None, [], None

    if pose_mask is None:
        return E, R, t, [], None

    pose_mask = pose_mask.ravel()

    pose_inlier_matches = [
        m for m, keep in zip(matches, pose_mask)
        if keep != 0
    ]

    return E, R, t, pose_inlier_matches, pose_mask

def main():
    # 先设置输入输出路径，并创建相应的文件夹
    image_folder = "image"
    output_folder = "output"

    keypoint_output_folder = Path(output_folder) / "keypoints"
    descriptor_output_folder = Path(output_folder) / "descriptors"
    match_output_folder = Path(output_folder) / "matches"

    create_folder(keypoint_output_folder)
    create_folder(descriptor_output_folder)
    create_folder(match_output_folder)

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
        print(f"[{index + 1}/{len(image_files)}] 正在处理：{image_path.name}")          #提示处理进度
        image, gray, keypoints, descriptors = extract_sift_features(image_path, sift)

        if image is None:
            continue

        print(f"检测到关键点数量：{len(keypoints)}")
        
        # 保存关键点可视化图片
        keypoint_save_path = keypoint_output_folder / f"{image_path.stem}_keypoints.jpg"
        save_keypoint_image(image, keypoints, keypoint_save_path)

        # 保存描述子
        if descriptors is not None:
            descriptor_save_path = descriptor_output_folder / f"{image_path.stem}_descriptors.npy"
            np.save(descriptor_save_path, descriptors)
        else:
            print(f"    警告：{image_path.name} 没有检测到描述子。")
        
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
    min_good_matches = 50  # 小于这个数量，说明两张图初步匹配关系较弱，直接跳过基础矩阵估计
    min_inlier_matches = 20 # 小于这个数量，说明两张图的几何关系不可靠，基础矩阵估计失败
    min_pose_inliers = 15 # 小于这个数量，说明姿态恢复的内点太少，结果不可靠
    # 基础矩阵 RANSAC 参数
    ransac_threshold = 3.0
    ransac_confidence = 0.99
    # 本质矩阵 RANSAC 参数
    essential_ransac_threshold = 1.0
    essential_confidence = 0.999

    total_pair_count = 0
    saved_match_count = 0 #统计总共处理的图片对数量和成功保存匹配图的数量
    pose_success_count = 0 # 统计成功恢复姿态的图片对数量
    match_log_path = Path(output_folder) / "match_log.csv" #保存匹配结果的表格文件路径
    pose_log_path = Path(output_folder) / "pose_log.csv" #保存姿态恢复结果的表格文件路径

    with open(match_log_path, "w", encoding="utf-8") as log_file, \
     open(pose_log_path, "w", encoding="utf-8") as pose_log_file:

        log_file.write("image1,image2,good_matches,fundamental_inliers,pose_inliers,saved,pose_success\n")

        pose_log_file.write(
            "image1,image2,"
            "r11,r12,r13,r21,r22,r23,r31,r32,r33,"
            "tx,ty,tz,"
            "euler_x_deg,euler_y_deg,euler_z_deg,"
            "good_matches,fundamental_inliers,pose_inliers\n")


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

                total_pair_count += 1

                # 第一步：SIFT 描述子匹配

                good_matches = match_two_images(
                    desc1,
                    desc2,
                )

                num_good_matches = len(good_matches)
                '''
                print(f"\n{name1} 与 {name2}")
                print(f"    SIFT ratio test 后匹配点数量：{num_good_matches}")
                '''
                if num_good_matches < min_good_matches:
                    log_file.write(f"{name1},{name2},{num_good_matches},0,False\n")
                    continue

                # 第二步：基础矩阵 F + RANSAC

                F, inlier_matches, F_mask = estimate_fundamental_matrix(
                    kp1,
                    kp2,
                    good_matches,
                    ransac_threshold=ransac_threshold,
                    confidence=ransac_confidence
                )

                num_inliers = len(inlier_matches)

                if F is None:
                    print("基础矩阵估计失败。")
                    log_file.write(f"{name1},{name2},{num_good_matches},0,False\n")
                    continue

                print(f"    基础矩阵 RANSAC 内点数量：{num_inliers}")

                if num_inliers < min_inlier_matches:  #判断内点个数是否达标
                    print(f"    几何内点少于 {min_inlier_matches}，不保存匹配图。")
                    log_file.write(f"{name1},{name2},{num_good_matches},{num_inliers},False\n")
                    continue

                # 第三步：保存几何内点匹配图

                match_save_path = match_output_folder / f"{name1}_to_{name2}_F_inliers.jpg"

                save_match_image(
                    img1,
                    kp1,
                    img2,
                    kp2,
                    inlier_matches,
                    match_save_path,
                    max_draw_matches=80
                )

                saved_match_count += 1
                print(f"    已保存几何内点匹配图：{match_save_path}")

                # 第四步：构造近似相机内参 K
                K = create_approx_camera_matrix(img1)

                # 第五步：构造本质矩阵，恢复相机姿态
                E, R, t, pose_inlier_matches, pose_mask = recover_camera_pose(
                    kp1,
                    kp2,
                    inlier_matches,
                    K,
                    ransac_threshold=essential_ransac_threshold,
                    confidence=essential_confidence)

                num_pose_inliers = len(pose_inlier_matches)

                if E is None or R is None or t is None:
                    print("    相机姿态恢复失败。")
                    log_file.write(
                        f"{name1},{name2},{num_good_matches},{num_inliers},0,True,False\n"
                    )
                    continue

                print(f"    姿态恢复有效点数量：{num_pose_inliers}")

                if num_pose_inliers < min_pose_inliers:
                    print(f"    姿态恢复有效点少于 {min_pose_inliers}，认为姿态不可靠。")
                    log_file.write(
                        f"{name1},{name2},{num_good_matches},{num_inliers},{num_pose_inliers},True,False\n"
                    )
                    continue

                pose_success_count += 1

                # 第六步：输出和保存姿态结果
                euler_angles = rotation_matrix_to_euler_angles(R)

                print("    姿态恢复成功。")
                print("    旋转矩阵 R：")
                print(R)
                print("    平移方向 t：")
                print(t)
                print("    欧拉角，单位为度：")
                print(euler_angles)

                R_flat = R.reshape(-1)
                t_flat = t.reshape(-1)

                pose_log_file.write(
                    f"{name1},{name2},"
                    f"{R_flat[0]},{R_flat[1]},{R_flat[2]},"
                    f"{R_flat[3]},{R_flat[4]},{R_flat[5]},"
                    f"{R_flat[6]},{R_flat[7]},{R_flat[8]},"
                    f"{t_flat[0]},{t_flat[1]},{t_flat[2]},"
                    f"{euler_angles[0]},{euler_angles[1]},{euler_angles[2]},"
                    f"{num_good_matches},{num_inliers},{num_pose_inliers}\n"
                )

                log_file.write(
                    f"{name1},{name2},{num_good_matches},{num_inliers},{num_pose_inliers},True,True\n"
                )

    print("\n全部处理完成。")
    print(f"总共比较图片组合数：{total_pair_count}")
    print(f"保存的有效几何匹配图数量：{saved_match_count}")
    print(f"成功恢复姿态的图片对数量：{pose_success_count}")
    print(f"匹配统计表已保存到：{match_log_path}")
    print(f"姿态结果表已保存到：{pose_log_path}")
    print(f"输出结果文件夹：{output_folder}")


if __name__ == "__main__":
    main()