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
    min_good_matches = 50  # 小于这个数量，说明两张图初步匹配关系较弱，直接跳过基础矩阵估计
    min_inlier_matches = 20 # 小于这个数量，说明两张图的几何关系不可靠，基础矩阵估计失败
    min_pose_inliers = 15 # 小于这个数量，说明姿态恢复的内点太少，结果不可靠
    # 基础矩阵 RANSAC 参数
    ransac_threshold = 3.0
    ransac_confidence = 0.99

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
                
            print(f"\n{name1} 与 {name2}")
            print(f"SIFT ratio test 后匹配点数量：{len(prime_matches)}")
                
            if len(prime_matches) < min_good_matches: #如果初步匹配点数量过少，说明两张图的内容差异较大，跳过后续步骤
                continue

    print("\n全部处理完成。")

if __name__ == "__main__":
    main()