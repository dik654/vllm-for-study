# Physical AI & 제조업 AI 가이드

## 목차
1. [개요](#개요)
2. [컴퓨터 비전 (품질 검사)](#컴퓨터-비전-품질-검사)
3. [로보틱스 & 자동화](#로보틱스--자동화)
4. [예지 보전 (Predictive Maintenance)](#예지-보전-predictive-maintenance)
5. [디지털 트윈](#디지털-트윈)
6. [공급망 최적화](#공급망-최적화)
7. [프로덕션 배포](#프로덕션-배포)

---

## 개요

### Physical AI란?
**정의**: 물리적 세계와 상호작용하는 AI 시스템

**제조업에서의 AI 적용 분야**:
1. **품질 검사** (Computer Vision)
2. **로봇 자동화** (Robotics)
3. **예지 보전** (Predictive Maintenance)
4. **디지털 트윈** (Digital Twin)
5. **공급망 최적화** (Supply Chain)
6. **생산 계획** (Production Planning)

### 2024-2025 제조업 AI 트렌드

```python
"""
제조업 AI 주요 트렌드

1. Foundation Models for Robotics
   - RT-2 (Google): VLM for robot control
   - PaLM-E: Embodied multimodal LLM

2. Visual Inspection at Scale
   - Anomaly Detection (Zero-shot)
   - 3D Defect Detection

3. Predictive Maintenance
   - Time-series Transformers
   - Anomaly Detection (Unsupervised)

4. Digital Twin + GenAI
   - Simulation with LLMs
   - 최적화 자동화

5. Edge AI
   - On-device inference (TensorRT, ONNX)
   - Federated Learning
"""
```

---

## 컴퓨터 비전 (품질 검사)

### 1. 결함 탐지 (Defect Detection)

**문제**: 제품에서 결함 찾기 (균열, 스크래치, 색상 불량 등)

```python
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image

class DefectDetector:
    """
    결함 탐지 시스템

    방법론:
    1. Supervised (라벨링된 데이터)
    2. Anomaly Detection (정상 데이터만)
    3. Few-shot Learning (적은 예시)
    """
    def __init__(self, method='supervised'):
        self.method = method

        if method == 'supervised':
            self.model = self.build_supervised_model()
        elif method == 'anomaly':
            self.model = self.build_anomaly_model()
        else:  # few-shot
            self.model = self.build_fewshot_model()

    # ========== Method 1: Supervised Learning ==========
    def build_supervised_model(self):
        """
        Supervised Defect Detection

        장점: 높은 정확도
        단점: 많은 라벨 필요

        아키텍처: ResNet50 + Classification Head
        """
        from torchvision.models import resnet50, ResNet50_Weights

        # Pre-trained ResNet50
        model = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Classification head 교체
        num_classes = 5  # [정상, 균열, 스크래치, 색상불량, 변형]
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        return model

    def train_supervised(self, train_loader, epochs=50):
        """Supervised 학습"""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.model.train()
        for epoch in range(epochs):
            for images, labels in train_loader:
                # Forward
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # ========== Method 2: Anomaly Detection ==========
    def build_anomaly_model(self):
        """
        Anomaly Detection (정상 데이터만 학습)

        장점: 라벨링 불필요, 새로운 결함 탐지
        단점: 정상 데이터 대표성 중요

        방법: PaDiM (Patch Distribution Modeling)
        """
        from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

        # Feature Extractor
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
        model.eval()

        # Layer outputs 추출 (multi-scale features)
        self.feature_layers = ['layer1', 'layer2', 'layer3']

        return model

    def train_anomaly(self, normal_images):
        """
        Anomaly Detection 학습

        핵심: 정상 이미지의 feature distribution 학습
        """
        # Step 1: Feature extraction
        features = self.extract_features(normal_images)

        # Step 2: Gaussian distribution fitting (per patch)
        # features: (N, H, W, C) → 각 (h, w) 위치마다 Gaussian
        self.means = {}
        self.covs = {}

        for layer in self.feature_layers:
            layer_features = features[layer]  # (N, C, H, W)
            N, C, H, W = layer_features.shape

            # Reshape: (N, C, H, W) → (H, W, N, C)
            feats = layer_features.permute(2, 3, 0, 1).reshape(H, W, N*C)

            # 각 (h, w) 위치의 평균/공분산
            self.means[layer] = feats.mean(dim=0)  # (H, W, C)
            self.covs[layer] = torch.cov(feats.T)  # (C, C)

    def extract_features(self, images):
        """Multi-scale feature extraction"""
        features = {}

        def hook_fn(layer_name):
            def hook(module, input, output):
                features[layer_name] = output
            return hook

        # Hook 등록
        hooks = []
        for name, module in self.model.named_modules():
            if name in self.feature_layers:
                hooks.append(module.register_forward_hook(hook_fn(name)))

        # Forward pass
        with torch.no_grad():
            _ = self.model(images)

        # Hook 제거
        for hook in hooks:
            hook.remove()

        return features

    def detect_anomaly(self, image):
        """
        Anomaly Detection 추론

        반환: anomaly map (이미지 위치별 이상도)
        """
        features = self.extract_features(image.unsqueeze(0))

        anomaly_maps = []

        for layer in self.feature_layers:
            feats = features[layer]  # (1, C, H, W)
            mean = self.means[layer]
            cov = self.covs[layer]

            # Mahalanobis distance (이상도 측정)
            # d = √((x - μ)ᵀ Σ⁻¹ (x - μ))
            cov_inv = torch.linalg.inv(cov)

            # 각 위치별 거리 계산
            anomaly_map = torch.zeros(feats.shape[2:])
            for h in range(feats.shape[2]):
                for w in range(feats.shape[3]):
                    feat_vec = feats[0, :, h, w] - mean[h, w]
                    dist = torch.sqrt(feat_vec @ cov_inv @ feat_vec)
                    anomaly_map[h, w] = dist

            anomaly_maps.append(anomaly_map)

        # Multi-scale anomaly maps 결합
        final_anomaly_map = sum(anomaly_maps) / len(anomaly_maps)

        return final_anomaly_map

    # ========== Method 3: Few-shot Learning ==========
    def build_fewshot_model(self):
        """
        Few-shot Defect Detection

        장점: 적은 예시로 학습 (예: 5장)
        방법: Siamese Network or Prototypical Network
        """
        class SiameseNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                from torchvision.models import resnet18, ResNet18_Weights

                # Backbone
                backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
                self.encoder = nn.Sequential(*list(backbone.children())[:-1])

                # Embedding head
                self.fc = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128)
                )

            def forward(self, x):
                # x: (batch, 3, H, W)
                features = self.encoder(x)  # (batch, 512, 1, 1)
                features = features.view(features.size(0), -1)
                embeddings = self.fc(features)  # (batch, 128)

                # L2 normalize
                embeddings = F.normalize(embeddings, p=2, dim=1)

                return embeddings

        return SiameseNetwork()

    def train_fewshot(self, support_set, query_set):
        """
        Few-shot Learning

        support_set: {class_name: [img1, img2, ...], ...}
        query_set: [(img, label), ...]
        """
        # Step 1: Support set embeddings (prototypes)
        prototypes = {}

        self.model.eval()
        with torch.no_grad():
            for class_name, images in support_set.items():
                embeddings = [self.model(img.unsqueeze(0)) for img in images]
                # Prototype = 평균 embedding
                prototype = torch.stack(embeddings).mean(dim=0)
                prototypes[class_name] = prototype

        # Step 2: Query classification
        for query_img, true_label in query_set:
            query_emb = self.model(query_img.unsqueeze(0))

            # 가장 가까운 prototype 찾기
            min_dist = float('inf')
            predicted_class = None

            for class_name, prototype in prototypes.items():
                dist = torch.norm(query_emb - prototype)
                if dist < min_dist:
                    min_dist = dist
                    predicted_class = class_name

            print(f"True: {true_label}, Predicted: {predicted_class}, Distance: {min_dist:.3f}")

# ========== 실전 사용 ==========

# Method 1: Supervised (라벨 많음)
detector_sup = DefectDetector(method='supervised')
# detector_sup.train_supervised(train_loader, epochs=50)

# Method 2: Anomaly Detection (정상만)
detector_anomaly = DefectDetector(method='anomaly')
# detector_anomaly.train_anomaly(normal_images)

# 추론
test_image = Image.open("product.jpg")
test_tensor = transforms.ToTensor()(test_image)

anomaly_map = detector_anomaly.detect_anomaly(test_tensor)

# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(test_image)
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(anomaly_map, cmap='jet')
plt.title('Anomaly Map')
plt.colorbar()

# Threshold 적용
threshold = anomaly_map.mean() + 2 * anomaly_map.std()
defect_mask = (anomaly_map > threshold).numpy()

plt.subplot(1, 3, 3)
plt.imshow(test_image)
plt.imshow(defect_mask, alpha=0.5, cmap='Reds')
plt.title('Detected Defects')

plt.show()

"""
실전 팁:

1. 데이터 증강:
   - 회전, 크롭, 색상 조정
   - Mixup, CutMix

2. Multi-scale 검사:
   - 다양한 해상도
   - 작은 결함 탐지

3. Edge Deployment:
   - TensorRT로 최적화
   - ONNX 변환
   - Quantization (INT8)

4. Active Learning:
   - 불확실한 샘플만 라벨링
   - 점진적 성능 향상
"""
```

---

### 2. 3D 검사 (3D Inspection)

```python
import open3d as o3d

class 3DDefectDetector:
    """
    3D Point Cloud 기반 결함 탐지

    입력: Depth camera, LiDAR, Structured Light
    출력: 3D 결함 위치 및 크기

    사용 사례:
    - 용접 품질 검사
    - 표면 거칠기 측정
    - 조립 정렬 검사
    """
    def __init__(self):
        pass

    def capture_3d(self, depth_image, intrinsics):
        """
        Depth 이미지 → 3D Point Cloud

        depth_image: (H, W) depth values
        intrinsics: 카메라 내부 파라미터
        """
        # Depth → Point Cloud
        pcd = o3d.geometry.PointCloud()

        height, width = depth_image.shape
        points = []

        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']

        for v in range(height):
            for u in range(width):
                z = depth_image[v, u]
                if z == 0:  # Invalid depth
                    continue

                # Pixel → 3D
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy

                points.append([x, y, z])

        pcd.points = o3d.utility.Vector3dVector(np.array(points))

        return pcd

    def detect_defects_3d(self, pcd, reference_pcd):
        """
        3D 결함 탐지

        방법: Point cloud registration + difference
        """
        # Step 1: Alignment (ICP)
        transformation = self.align_point_clouds(pcd, reference_pcd)

        pcd_aligned = pcd.transform(transformation)

        # Step 2: Point-to-plane distance
        distances = pcd_aligned.compute_point_cloud_distance(reference_pcd)
        distances = np.asarray(distances)

        # Step 3: Outliers = Defects
        threshold = np.mean(distances) + 3 * np.std(distances)
        defect_indices = np.where(distances > threshold)[0]

        # 결함 point cloud
        defect_pcd = pcd_aligned.select_by_index(defect_indices)

        return defect_pcd, distances

    def align_point_clouds(self, source, target):
        """ICP (Iterative Closest Point) Registration"""
        threshold = 0.02  # 2mm

        # ICP
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold,
            np.eye(4),  # Initial transformation
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        return reg_p2p.transformation

    def measure_surface_roughness(self, pcd):
        """
        표면 거칠기 측정

        방법: Local surface normal variance
        """
        # Normal 추정
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30
        ))

        # KD-tree 구축
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        roughness_values = []

        for i in range(len(pcd.points)):
            # K-nearest neighbors
            [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], 30)

            # Neighbor normals
            neighbor_normals = np.asarray(pcd.normals)[idx, :]

            # Normal variance = roughness
            normal_variance = np.var(neighbor_normals, axis=0).sum()
            roughness_values.append(normal_variance)

        return np.array(roughness_values)

# 실전 사용
detector_3d = 3DDefectDetector()

# Depth 이미지 로드 (예시)
depth_img = cv2.imread("depth.png", cv2.IMREAD_ANYDEPTH)

intrinsics = {
    'fx': 525.0,
    'fy': 525.0,
    'cx': 320.0,
    'cy': 240.0
}

# Point cloud 생성
pcd = detector_3d.capture_3d(depth_img, intrinsics)

# Reference (정상 제품)
reference_pcd = o3d.io.read_point_cloud("reference.pcd")

# 결함 탐지
defect_pcd, distances = detector_3d.detect_defects_3d(pcd, reference_pcd)

# 시각화
o3d.visualization.draw_geometries([pcd, defect_pcd])
```

---

## 로보틱스 & 자동화

### 1. Robot Control with Vision-Language Models

```python
class RobotController:
    """
    VLM 기반 로봇 제어

    방법: RT-2 (Robotics Transformer 2)
    - 자연어 명령 → 로봇 액션
    - Vision + Language → Control

    예: "빨간 블록을 파란 상자에 넣어"
    """
    def __init__(self):
        # VLM for robotics (예: RT-2, PaLM-E)
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.model = AutoModelForVision2Seq.from_pretrained("google/rt-2-base")
        self.processor = AutoProcessor.from_pretrained("google/rt-2-base")

    def parse_command(self, image, text_command):
        """
        자연어 + 이미지 → 로봇 액션

        출력: [dx, dy, dz, roll, pitch, yaw, gripper]
        """
        # 입력 처리
        inputs = self.processor(
            text=text_command,
            images=image,
            return_tensors="pt"
        )

        # VLM 추론
        outputs = self.model.generate(**inputs)

        # 디코딩: 액션 시퀀스
        action_tokens = self.processor.decode(outputs[0])

        # 파싱: 토큰 → 로봇 제어 신호
        actions = self.parse_action_tokens(action_tokens)

        return actions

    def parse_action_tokens(self, tokens):
        """
        Action tokens → Robot control signals

        RT-2 format:
        "move[dx=0.1, dy=0.0, dz=-0.05, gripper=close]"
        """
        import re

        # 정규식으로 파싱
        pattern = r'(\w+)\[(.*?)\]'
        matches = re.findall(pattern, tokens)

        actions = []

        for action_type, params in matches:
            param_dict = {}
            for param in params.split(','):
                key, value = param.split('=')
                param_dict[key.strip()] = float(value.strip()) if value.replace('.', '').isdigit() else value.strip()

            actions.append({
                'type': action_type,
                'params': param_dict
            })

        return actions

    def execute_action(self, action, robot_api):
        """
        로봇 API로 액션 실행

        robot_api: ROS, MoveIt 등
        """
        if action['type'] == 'move':
            dx = action['params']['dx']
            dy = action['params']['dy']
            dz = action['params']['dz']

            # 현재 위치
            current_pose = robot_api.get_current_pose()

            # 목표 위치
            target_pose = current_pose.copy()
            target_pose[0] += dx
            target_pose[1] += dy
            target_pose[2] += dz

            # 이동
            robot_api.move_to(target_pose)

        elif action['type'] == 'grasp':
            gripper_state = action['params']['gripper']
            if gripper_state == 'close':
                robot_api.close_gripper()
            else:
                robot_api.open_gripper()

# 실전 사용
robot = RobotController()

# 카메라 이미지
camera_image = Image.open("workspace.jpg")

# 자연어 명령
command = "빨간 블록을 집어서 왼쪽 상자에 넣어"

# VLM 추론
actions = robot.parse_command(camera_image, command)

print("Planned Actions:")
for i, action in enumerate(actions):
    print(f"Step {i+1}: {action}")

# 출력:
# Step 1: {'type': 'move', 'params': {'dx': -0.15, 'dy': 0.05, 'dz': -0.10}}
# Step 2: {'type': 'grasp', 'params': {'gripper': 'close'}}
# Step 3: {'type': 'move', 'params': {'dx': 0.30, 'dy': 0.00, 'dz': 0.10}}
# Step 4: {'type': 'grasp', 'params': {'gripper': 'open'}}

# 실행 (ROS 등)
# for action in actions:
#     robot.execute_action(action, robot_api)
```

---

### 2. Bin Picking with 6D Pose Estimation

```python
class BinPicking:
    """
    Bin Picking: 무작위로 쌓인 부품 집기

    핵심: 6D Pose Estimation
    - 위치 (x, y, z)
    - 방향 (roll, pitch, yaw)

    방법: Deep Learning (PointNet++, PVN3D)
    """
    def __init__(self):
        # 6D Pose Estimation 모델
        self.pose_model = self.load_pose_model()

    def load_pose_model(self):
        """
        6D Pose Estimation Model

        입력: RGB-D 이미지
        출력: 물체별 6D pose
        """
        # 예: PVN3D, DenseFusion 등
        # 여기서는 간략화
        pass

    def estimate_6d_pose(self, rgb_image, depth_image, object_models):
        """
        6D Pose Estimation

        object_models: CAD 모델들 (3D mesh)
        """
        # Point cloud 생성
        pcd = self.rgbd_to_pointcloud(rgb_image, depth_image)

        # Instance segmentation (각 물체 분리)
        instances = self.segment_instances(pcd)

        poses = []

        for instance_pcd in instances:
            # 각 instance의 pose 추정
            best_pose = None
            best_score = -float('inf')

            for obj_name, obj_model in object_models.items():
                # Template matching in 3D
                pose, score = self.match_template_3d(instance_pcd, obj_model)

                if score > best_score:
                    best_score = score
                    best_pose = {
                        'object': obj_name,
                        'translation': pose[:3],
                        'rotation': pose[3:],
                        'confidence': score
                    }

            poses.append(best_pose)

        return poses

    def match_template_3d(self, instance_pcd, template_mesh):
        """
        3D Template Matching

        방법: ICP + RANSAC
        """
        # Template → Point cloud
        template_pcd = template_mesh.sample_points_uniformly(number_of_points=1000)

        # RANSAC + ICP
        transformation, score = self.ransac_icp(instance_pcd, template_pcd)

        # Transformation → (translation, rotation)
        translation = transformation[:3, 3]
        rotation = self.matrix_to_euler(transformation[:3, :3])

        pose = np.concatenate([translation, rotation])

        return pose, score

    def plan_grasp(self, pose, object_model):
        """
        Grasp Planning

        입력: 물체의 6D pose
        출력: Gripper approach pose
        """
        # Grasp pose candidates 생성
        grasp_candidates = self.generate_grasp_candidates(object_model)

        # 각 후보를 world coordinate로 변환
        world_grasps = []
        for grasp in grasp_candidates:
            # Object frame → World frame
            world_grasp = self.transform_grasp(grasp, pose)
            world_grasps.append(world_grasp)

        # 충돌 검사 & 최적 grasp 선택
        valid_grasps = [g for g in world_grasps if self.is_collision_free(g)]

        if not valid_grasps:
            return None

        # 가장 점수 높은 grasp
        best_grasp = max(valid_grasps, key=lambda g: g['score'])

        return best_grasp

    def generate_grasp_candidates(self, object_model):
        """
        Grasp Candidate 생성

        방법: Antipodal grasp sampling
        """
        candidates = []

        # Object surface points
        surface_points = object_model.sample_points_uniformly(1000)

        for i in range(len(surface_points.points)):
            for j in range(i+1, len(surface_points.points)):
                p1 = surface_points.points[i]
                p2 = surface_points.points[j]

                # 두 점 사이 거리 (그리퍼 폭)
                distance = np.linalg.norm(p1 - p2)

                if 0.02 < distance < 0.08:  # 그리퍼 범위
                    # Grasp pose
                    grasp_center = (p1 + p2) / 2
                    grasp_direction = (p2 - p1) / distance

                    # Score (간단히: 중심에 가까울수록 높음)
                    score = 1.0 / (1.0 + np.linalg.norm(grasp_center))

                    candidates.append({
                        'position': grasp_center,
                        'direction': grasp_direction,
                        'width': distance,
                        'score': score
                    })

        return candidates

# 실전 사용
bin_picker = BinPicking()

# RGB-D 이미지
rgb = cv2.imread("bin_rgb.jpg")
depth = cv2.imread("bin_depth.png", cv2.IMREAD_ANYDEPTH)

# Object models (CAD)
object_models = {
    'gear': o3d.io.read_triangle_mesh("gear.stl"),
    'bolt': o3d.io.read_triangle_mesh("bolt.stl"),
}

# 6D Pose Estimation
poses = bin_picker.estimate_6d_pose(rgb, depth, object_models)

print("Detected Objects:")
for pose in poses:
    print(f"  {pose['object']}: pos={pose['translation']}, rot={pose['rotation']}, conf={pose['confidence']:.2f}")

# Grasp Planning
for pose in poses:
    grasp = bin_picker.plan_grasp(pose, object_models[pose['object']])
    if grasp:
        print(f"Grasp {pose['object']}: {grasp['position']}")
```

---

## 예지 보전 (Predictive Maintenance)

### 1. Time-Series Anomaly Detection

```python
class PredictiveMaintenance:
    """
    예지 보전: 설비 고장 예측

    입력: 센서 데이터 (진동, 온도, 압력, 전류 등)
    출력: 고장 확률, 잔여 수명 (RUL)

    방법:
    1. Anomaly Detection (비지도)
    2. RUL Prediction (지도)
    """
    def __init__(self, method='transformer'):
        self.method = method

        if method == 'transformer':
            self.model = self.build_transformer_model()
        else:  # 'lstm'
            self.model = self.build_lstm_model()

    # ========== Transformer for Time-Series ==========
    def build_transformer_model(self):
        """
        Time-Series Transformer

        장점: Long-range dependencies
        아키텍처: Informer, Autoformer
        """
        class TimeSeriesTransformer(nn.Module):
            def __init__(self, input_dim=10, d_model=128, nhead=8, num_layers=4):
                super().__init__()

                # Input embedding
                self.embedding = nn.Linear(input_dim, d_model)

                # Positional encoding
                self.pos_encoder = PositionalEncoding(d_model)

                # Transformer encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=512,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

                # Output heads
                self.anomaly_head = nn.Linear(d_model, 1)  # Anomaly score
                self.rul_head = nn.Linear(d_model, 1)  # RUL (Remaining Useful Life)

            def forward(self, x):
                # x: (batch, seq_len, input_dim)

                # Embedding
                x = self.embedding(x)  # (batch, seq_len, d_model)

                # Positional encoding
                x = self.pos_encoder(x)

                # Transformer
                x = self.transformer(x)  # (batch, seq_len, d_model)

                # Last timestep
                x_last = x[:, -1, :]  # (batch, d_model)

                # Predictions
                anomaly_score = torch.sigmoid(self.anomaly_head(x_last))
                rul = F.relu(self.rul_head(x_last))  # RUL >= 0

                return anomaly_score, rul

        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=5000):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len).unsqueeze(1).float()
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)

                self.register_buffer('pe', pe.unsqueeze(0))

            def forward(self, x):
                return x + self.pe[:, :x.size(1)]

        return TimeSeriesTransformer()

    def train_model(self, train_loader, epochs=100):
        """학습"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        # Loss functions
        anomaly_criterion = nn.BCELoss()
        rul_criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                sensor_data, anomaly_label, rul_target = batch

                # Forward
                anomaly_score, rul_pred = self.model(sensor_data)

                # Loss
                loss_anomaly = anomaly_criterion(anomaly_score.squeeze(), anomaly_label)
                loss_rul = rul_criterion(rul_pred.squeeze(), rul_target)

                loss = loss_anomaly + 0.5 * loss_rul

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    def predict_maintenance(self, sensor_sequence):
        """
        예지 보전 예측

        sensor_sequence: (seq_len, num_sensors)
        """
        self.model.eval()
        with torch.no_grad():
            anomaly_score, rul = self.model(sensor_sequence.unsqueeze(0))

        anomaly_score = anomaly_score.item()
        rul = rul.item()

        # 판단
        if anomaly_score > 0.7:
            status = "고장 위험"
            recommendation = f"즉시 점검 필요 (예상 잔여 수명: {rul:.1f}시간)"
        elif anomaly_score > 0.4:
            status = "주의"
            recommendation = f"모니터링 강화 (예상 잔여 수명: {rul:.1f}시간)"
        else:
            status = "정상"
            recommendation = f"정상 운영 (예상 잔여 수명: {rul:.1f}시간)"

        return {
            'status': status,
            'anomaly_score': anomaly_score,
            'rul_hours': rul,
            'recommendation': recommendation
        }

# 실전 사용
pm = PredictiveMaintenance(method='transformer')

# 센서 데이터 (예시)
# 10개 센서 × 100 timesteps
sensor_data = np.random.randn(100, 10)  # [진동, 온도, 압력, ...]

# 예측
result = pm.predict_maintenance(torch.FloatTensor(sensor_data))

print(f"상태: {result['status']}")
print(f"이상도: {result['anomaly_score']:.2%}")
print(f"잔여 수명: {result['rul_hours']:.1f}시간")
print(f"권장사항: {result['recommendation']}")
```

---

## 디지털 트윈

### 1. Simulation with LLMs

```python
class DigitalTwin:
    """
    Digital Twin + GenAI

    핵심: LLM으로 시뮬레이션 제어 및 최적화

    사용 사례:
    - 생산 라인 최적화
    - What-if 시나리오 분석
    - 자동 파라미터 튜닝
    """
    def __init__(self, simulator):
        self.simulator = simulator  # Physics engine (PyBullet, MuJoCo 등)
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

        # 시뮬레이션 히스토리
        self.history = []

    def natural_language_control(self, user_query):
        """
        자연어 → 시뮬레이션 제어

        예: "컨베이어 속도를 20% 증가시켜"
        """
        # LLM으로 파라미터 추출
        prompt = f"""
        사용자 요청: {user_query}

        현재 시뮬레이션 상태:
        - 컨베이어 속도: {self.simulator.conveyor_speed} m/s
        - 로봇 팔 속도: {self.simulator.robot_speed} m/s
        - 생산량: {self.simulator.throughput} units/hour

        요청을 시뮬레이션 파라미터 변경으로 변환하세요.

        JSON 형식:
        {{
            "parameters": {{
                "conveyor_speed": 1.2,
                "robot_speed": 0.5
            }},
            "reasoning": "..."
        }}
        """

        response = self.llm.invoke(prompt)
        change = json.loads(response.content)

        # 파라미터 적용
        for param, value in change['parameters'].items():
            setattr(self.simulator, param, value)

        # 시뮬레이션 실행
        result = self.simulator.run(duration=3600)  # 1시간

        # 히스토리 저장
        self.history.append({
            'query': user_query,
            'changes': change,
            'result': result
        })

        return result

    def optimize_with_llm(self, objective):
        """
        LLM 기반 최적화

        objective: "생산량 최대화", "에너지 최소화" 등
        """
        # Initial state
        baseline = self.simulator.run(duration=3600)

        best_config = self.simulator.get_config()
        best_score = self.evaluate_objective(baseline, objective)

        # Iterative optimization
        for iteration in range(10):
            # LLM에게 개선 방안 문의
            prompt = f"""
            목표: {objective}

            현재 설정:
            {json.dumps(self.simulator.get_config(), indent=2)}

            현재 성능:
            - 생산량: {baseline['throughput']}
            - 에너지: {baseline['energy']}
            - 불량률: {baseline['defect_rate']}

            목표 달성을 위한 파라미터 변경을 제안하세요.
            JSON 형식으로 반환.
            """

            response = self.llm.invoke(prompt)
            suggestion = json.loads(response.content)

            # 제안 적용 및 시뮬레이션
            self.simulator.set_config(suggestion['parameters'])
            result = self.simulator.run(duration=3600)

            # 평가
            score = self.evaluate_objective(result, objective)

            if score > best_score:
                best_score = score
                best_config = suggestion['parameters']
                print(f"Iteration {iteration+1}: 개선됨! Score: {score:.2f}")
            else:
                print(f"Iteration {iteration+1}: 개선 없음")

        # 최적 설정 적용
        self.simulator.set_config(best_config)

        return best_config, best_score

    def evaluate_objective(self, result, objective):
        """목표 함수 평가"""
        if objective == "생산량 최대화":
            return result['throughput']
        elif objective == "에너지 최소화":
            return -result['energy']
        elif objective == "품질 최대화":
            return 1 - result['defect_rate']
        else:
            # Multi-objective (예: 생산량 / 에너지)
            return result['throughput'] / result['energy']

# 실전 사용
# simulator = ProductionLineSimulator()  # PyBullet, etc.
# twin = DigitalTwin(simulator)

# 자연어 제어
# result = twin.natural_language_control("생산량을 30% 증가시키되 에너지는 10% 이하로 증가")

# LLM 최적화
# best_config, score = twin.optimize_with_llm("생산량 최대화")
```

---

## 공급망 최적화

### 1. Demand Forecasting

```python
class SupplyChainOptimizer:
    """
    공급망 최적화

    1. 수요 예측 (Demand Forecasting)
    2. 재고 최적화 (Inventory Optimization)
    3. 물류 최적화 (Logistics)
    """
    def __init__(self):
        self.forecast_model = self.build_forecast_model()

    def build_forecast_model(self):
        """
        Time-series Forecasting

        방법: Temporal Fusion Transformer (TFT)
        """
        from pytorch_forecasting import TemporalFusionTransformer

        # TFT 설정
        model = TemporalFusionTransformer.from_dataset(
            training_data,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=7,  # 7 quantiles
            loss=QuantileLoss(),
        )

        return model

    def forecast_demand(self, historical_data, horizon=30):
        """
        수요 예측

        historical_data: (timesteps, features)
        horizon: 예측 기간 (days)

        features: [sales, price, promotion, seasonality, ...]
        """
        # 예측
        predictions = self.forecast_model.predict(
            historical_data,
            mode="quantiles",
            return_x=True
        )

        # Quantile 예측 (uncertainty)
        forecast = {
            'mean': predictions.output.quantile(0.5),
            'lower_bound': predictions.output.quantile(0.1),  # 90% CI
            'upper_bound': predictions.output.quantile(0.9),
        }

        return forecast

    def optimize_inventory(self, forecast, current_stock, lead_time=7):
        """
        재고 최적화

        방법: (s, S) policy
        - s: Reorder point
        - S: Order-up-to level
        """
        # Safety stock 계산
        forecast_std = (forecast['upper_bound'] - forecast['lower_bound']) / 2.6  # 90% CI → std

        # Service level 99%
        z_score = 2.33  # 99% service level

        safety_stock = z_score * forecast_std * np.sqrt(lead_time)

        # Reorder point
        reorder_point = forecast['mean'] * lead_time + safety_stock

        # Order-up-to level
        review_period = 7  # days
        order_up_to = forecast['mean'] * (lead_time + review_period) + safety_stock

        # 주문 판단
        if current_stock < reorder_point:
            order_quantity = order_up_to - current_stock
            return {
                'action': 'order',
                'quantity': order_quantity,
                'reason': f"재고({current_stock:.0f}) < 재주문점({reorder_point:.0f})"
            }
        else:
            return {
                'action': 'wait',
                'quantity': 0,
                'reason': f"재고({current_stock:.0f}) 충분"
            }

# 실전 사용
optimizer = SupplyChainOptimizer()

# 과거 판매 데이터
historical_sales = np.array([...])  # 90일 데이터

# 수요 예측
forecast = optimizer.forecast_demand(historical_sales, horizon=30)

print("30일 수요 예측:")
print(f"  평균: {forecast['mean']:.0f}")
print(f"  범위: {forecast['lower_bound']:.0f} ~ {forecast['upper_bound']:.0f}")

# 재고 최적화
current_stock = 500
decision = optimizer.optimize_inventory(forecast, current_stock)

print(f"\n재고 결정: {decision['action']}")
if decision['action'] == 'order':
    print(f"  주문량: {decision['quantity']:.0f}")
print(f"  이유: {decision['reason']}")
```

---

## 프로덕션 배포

### 1. Edge Deployment

```python
class EdgeDeployment:
    """
    Edge AI 배포 (공장 현장)

    요구사항:
    - 낮은 레이턴시 (< 100ms)
    - 제한된 하드웨어 (Jetson, RaspberryPi)
    - 오프라인 동작

    최적화:
    - TensorRT, ONNX
    - Quantization (INT8)
    - Model Pruning
    """
    def optimize_for_edge(self, model_path):
        """
        PyTorch → ONNX → TensorRT
        """
        import torch
        import onnx
        from onnxruntime import InferenceSession

        # Step 1: PyTorch → ONNX
        model = torch.load(model_path)
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)

        torch.onnx.export(
            model,
            dummy_input,
            "model.onnx",
            export_params=True,
            opset_version=13,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        # Step 2: ONNX 검증
        onnx_model = onnx.load("model.onnx")
        onnx.checker.check_model(onnx_model)

        # Step 3: ONNX Runtime (CPU/GPU)
        session = InferenceSession("model.onnx", providers=['CUDAExecutionProvider'])

        return session

    def quantize_model(self, onnx_model_path):
        """
        Quantization: FP32 → INT8

        장점: 4배 메모리 절감, 2-4배 속도 향상
        """
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantized_model_path = "model_quantized.onnx"

        quantize_dynamic(
            onnx_model_path,
            quantized_model_path,
            weight_type=QuantType.QUInt8
        )

        return quantized_model_path

    def benchmark(self, session, num_runs=100):
        """성능 벤치마크"""
        import time

        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

        # Warmup
        for _ in range(10):
            _ = session.run(None, {'input': dummy_input})

        # Benchmark
        start = time.time()
        for _ in range(num_runs):
            _ = session.run(None, {'input': dummy_input})
        elapsed = time.time() - start

        latency = elapsed / num_runs * 1000  # ms

        print(f"평균 레이턴시: {latency:.2f}ms")
        print(f"처리량: {1000 / latency:.1f} FPS")

        return latency

# 실전 사용
deployer = EdgeDeployment()

# 최적화
session = deployer.optimize_for_edge("defect_detector.pth")

# Quantization
quantized_path = deployer.quantize_model("model.onnx")
quantized_session = InferenceSession(quantized_path)

# 벤치마크
print("FP32 모델:")
latency_fp32 = deployer.benchmark(session)

print("\nINT8 모델:")
latency_int8 = deployer.benchmark(quantized_session)

print(f"\n속도 향상: {latency_fp32 / latency_int8:.1f}배")
```

---

## 핵심 요약

### 제조업 AI 적용 분야

| 분야 | 기술 | ROI | 난이도 |
|------|------|-----|--------|
| 품질 검사 | Computer Vision (Anomaly Detection) | 높음 | 중간 |
| 로봇 자동화 | VLM, 6D Pose Estimation | 매우 높음 | 높음 |
| 예지 보전 | Time-Series Transformer | 높음 | 중간 |
| 디지털 트윈 | Simulation + LLM | 중간 | 높음 |
| 공급망 최적화 | Forecasting, Optimization | 높음 | 중간 |

### 성공 요인

1. **데이터 품질**
   - 라벨링 정확도
   - 다양한 시나리오 커버

2. **Edge 최적화**
   - 레이턴시 < 100ms
   - Quantization, TensorRT

3. **Human-in-the-Loop**
   - 불확실한 경우 사람 개입
   - 지속적 학습

4. **ROI 측정**
   - 불량률 감소: X%
   - 다운타임 감소: Y시간
   - 비용 절감: $Z

---

**작성일**: 2024-11-18
**업데이트**: Phase 6 - Current Trends
