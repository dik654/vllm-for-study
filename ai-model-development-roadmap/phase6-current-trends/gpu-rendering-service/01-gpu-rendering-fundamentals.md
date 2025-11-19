# GPU 렌더링 기초 이론

## 목차
1. [렌더링 기초 개념](#렌더링-기초-개념)
2. [CPU vs GPU 렌더링](#cpu-vs-gpu-렌더링)
3. [GPU 렌더링 파이프라인](#gpu-렌더링-파이프라인)
4. [레이트레이싱과 래스터화](#레이트레이싱과-래스터화)
5. [RTX 아키텍처의 혁신](#rtx-아키텍처의-혁신)
6. [렌더링 최적화 기법](#렌더링-최적화-기법)

---

## 렌더링 기초 개념

### 렌더링이란?

**정의**: 3D 모델, 텍스처, 조명 정보를 최종 2D 이미지로 변환하는 과정

```
3D Scene (입력)          Rendering          2D Image (출력)
- 모델 (Mesh)      →                  →    최종 이미지
- 텍스처                                     (PNG, EXR 등)
- 조명
- 카메라
- 머티리얼
```

### 렌더링의 종류

#### 1. 실시간 렌더링 (Real-time Rendering)
**용도**: 게임, VR/AR, 인터랙티브 애플리케이션

```python
# 실시간 렌더링 목표
FPS = 60  # 초당 60프레임
frame_time = 1000 / FPS  # 16.67ms per frame

# 제약사항
max_render_time_per_frame = 16.67  # milliseconds
```

**특징**:
- 초당 30-120 프레임 목표
- 품질보다 속도 우선
- 근사 기법 사용 (Rasterization)
- 실시간 인터랙션 필요

#### 2. 오프라인 렌더링 (Offline Rendering)
**용도**: 영화, 애니메이션, 광고, 건축 시각화

```python
# 오프라인 렌더링 (영화 품질)
frame_resolution = (3840, 2160)  # 4K
samples_per_pixel = 1000  # 노이즈 제거
render_time_per_frame = 30 * 60  # 30분 per frame

# 24fps 영화 1분 = 1,440프레임
total_render_time = 1440 * 30  # 43,200분 = 30일 (단일 GPU)
```

**특징**:
- 프레임당 수 분 ~ 수 시간
- 최고 품질 추구
- 물리 기반 렌더링 (Physically Based Rendering)
- 정확한 광학 시뮬레이션

---

## CPU vs GPU 렌더링

### 아키텍처 비교

```
CPU (예: Intel Xeon)
┌─────────────────────────────┐
│ Core 1  Core 2  Core 3  ... │  8-64 코어
│ (강력)  (강력)  (강력)      │  높은 클럭
│ 복잡한 로직 처리 가능        │  큰 캐시
└─────────────────────────────┘

GPU (예: RTX A6000)
┌─────────────────────────────┐
│ ████████████████████████... │  10,752 CUDA 코어
│ (약함) (약함) (약함) ...    │  낮은 클럭
│ 단순 병렬 작업 특화          │  작은 캐시
└─────────────────────────────┘
```

### 성능 비교

| 특성 | CPU 렌더링 | GPU 렌더링 |
|------|-----------|-----------|
| **병렬성** | 8-64 스레드 | 10,000+ 스레드 |
| **렌더 속도** | 1x (기준) | 10-50x |
| **메모리** | 128-512GB | 24-48GB |
| **적합 작업** | 복잡한 셰이더, 큰 씬 | 단순 반복, 병렬 가능 |
| **비용** | 저렴 | 고가 |

### 실제 벤치마크 예시

```python
# Blender Cycles 렌더링 벤치마크
scene = "BMW Benchmark Scene"
resolution = (1920, 1080)
samples = 100

render_times = {
    "Intel Xeon 16-core": "5m 42s",
    "AMD Threadripper 32-core": "3m 15s",
    "NVIDIA RTX 3090": "0m 45s",
    "NVIDIA RTX A6000": "0m 38s",  # 48GB 메모리로 더 큰 씬 가능
}

# 속도 향상
speedup_rtx_a6000 = 342 / 38  # 9배 빠름
```

---

## GPU 렌더링 파이프라인

### 전체 파이프라인

```
┌──────────────────────────────────────────────────────────┐
│                  GPU 렌더링 파이프라인                     │
└──────────────────────────────────────────────────────────┘

1. Scene Setup
   └─→ 3D 모델 로드
   └─→ 텍스처 로드
   └─→ 머티리얼 설정
   └─→ 조명 배치

2. Data Transfer (CPU → GPU)
   └─→ 버텍스 데이터 → VRAM
   └─→ 텍스처 → VRAM
   └─→ 셰이더 코드 컴파일

3. Vertex Processing
   └─→ Vertex Shader 실행
   └─→ 3D 좌표 변환 (World → Screen)

4. Rasterization / Ray Tracing
   ┌─→ [Rasterization Path]
   │   └─→ 삼각형을 픽셀로 변환
   │   └─→ Fragment Shader 실행
   │
   └─→ [Ray Tracing Path]
       └─→ RT Core에서 광선 추적
       └─→ BVH 가속 구조 사용
       └─→ 교차점 계산

5. Shading
   └─→ 조명 계산
   └─→ 텍스처 샘플링
   └─→ 머티리얼 평가

6. Post-Processing
   └─→ Denoising (AI Tensor Core)
   └─→ Bloom, DOF, Motion Blur
   └─→ Tone Mapping

7. Output
   └─→ Frame Buffer → CPU
   └─→ 이미지 저장
```

### CUDA 기반 렌더링 코드 예시

```cpp
// CUDA 커널: 각 픽셀마다 병렬로 레이 트레이싱
__global__ void render_kernel(
    float3* output_image,
    int width,
    int height,
    Camera camera,
    Scene* scene
) {
    // 픽셀 좌표 계산
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // 각 픽셀은 독립적으로 처리됨 (완벽한 병렬화!)
    int pixel_index = y * width + x;

    // 카메라에서 픽셀로 광선 생성
    Ray ray = camera.generate_ray(x, y);

    // 레이 트레이싱 수행
    float3 color = trace_ray(ray, scene, 0);

    // 결과 저장
    output_image[pixel_index] = color;
}

// 호스트 코드
void render_scene() {
    // GPU 메모리 할당
    float3* d_output;
    cudaMalloc(&d_output, width * height * sizeof(float3));

    // 블록/그리드 설정
    dim3 block(16, 16);  // 256 스레드 per block
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    // 커널 실행 - 수천 개 픽셀 동시 처리!
    render_kernel<<<grid, block>>>(
        d_output, width, height, camera, scene
    );

    // 결과 복사
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
}
```

---

## 레이트레이싱과 래스터화

### 래스터화 (Rasterization)

**원리**: 삼각형을 픽셀로 변환 (근사 기법)

```
3D Triangle         →        Screen Pixels
    /\                         ██
   /  \                       ████
  /____\                     ██████

장점:
✓ 매우 빠름 (실시간)
✓ GPU 하드웨어 최적화
✓ 예측 가능한 성능

단점:
✗ 정확한 반사/굴절 어려움
✗ 부드러운 그림자 어려움
✗ 간접 조명 근사만 가능
```

```glsl
// Fragment Shader (래스터화)
#version 450

in vec3 fragPosition;
in vec3 fragNormal;
in vec2 fragTexCoord;

uniform sampler2D diffuseTexture;
uniform vec3 lightPosition;

out vec4 fragColor;

void main() {
    // 간단한 조명 계산 (Phong)
    vec3 normal = normalize(fragNormal);
    vec3 lightDir = normalize(lightPosition - fragPosition);
    float diffuse = max(dot(normal, lightDir), 0.0);

    vec3 texColor = texture(diffuseTexture, fragTexCoord).rgb;
    fragColor = vec4(texColor * diffuse, 1.0);

    // 문제: 반사는? 굴절은? 부드러운 그림자는?
    // → 근사 기법으로만 구현 가능
}
```

### 레이트레이싱 (Ray Tracing)

**원리**: 실제 빛의 경로를 시뮬레이션 (물리 기반)

```
Camera               Scene                   Result
  📷 ─────→ 🎯 반사 ─→ 💡                   정확한 반사
            │                                정확한 굴절
            └─→ 🌐 굴절 ─→ 💡               부드러운 그림자
                                             글로벌 일루미네이션

장점:
✓ 물리적으로 정확
✓ 자연스러운 반사/굴절
✓ 부드러운 그림자
✓ 글로벌 일루미네이션

단점:
✗ 계산량 매우 많음
✗ 전통적으로 실시간 불가능
✗ 메모리 많이 사용
```

```cpp
// Path Tracing (물리 기반 레이 트레이싱)
__device__ float3 trace_ray(Ray ray, Scene* scene, int depth) {
    if (depth > MAX_DEPTH) return float3(0, 0, 0);

    // 씬과의 교차점 찾기
    HitRecord hit;
    if (!scene->intersect(ray, hit)) {
        return scene->environment_map(ray.direction);
    }

    // 머티리얼 평가
    Material* mat = hit.material;

    // Bidirectional Reflectance Distribution Function
    float3 scattered_direction;
    float3 attenuation;
    mat->scatter(ray, hit, scattered_direction, attenuation);

    // 재귀적으로 추적 (반사/굴절)
    Ray scattered_ray(hit.position, scattered_direction);
    float3 incoming_light = trace_ray(scattered_ray, scene, depth + 1);

    // 최종 색상 = 머티리얼 반사율 × 들어오는 빛
    return hit.emission + attenuation * incoming_light;
}
```

---

## RTX 아키텍처의 혁신

### RTX GPU 구조 (RTX A6000 기준)

```
┌─────────────────────────────────────────────────────────┐
│                   RTX A6000 GPU                          │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ CUDA Cores   │  │  RT Cores    │  │ Tensor Cores │  │
│  │ (10,752개)   │  │  (84개)      │  │  (336개)     │  │
│  │              │  │              │  │              │  │
│  │ 일반 계산    │  │ 레이 트레이싱│  │ AI/Denoising │  │
│  │ 셰이딩       │  │ BVH 가속     │  │ DLSS         │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                           │
│  ┌───────────────────────────────────────────────────┐  │
│  │           VRAM: 48GB GDDR6 ECC                    │  │
│  │           Bandwidth: 768 GB/s                     │  │
│  └───────────────────────────────────────────────────┘  │
│                                                           │
│  ┌───────────────────────────────────────────────────┐  │
│  │           NVLink: 112.5 GB/s (멀티 GPU)           │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 1. RT Core (Ray Tracing Core)

**기능**: 하드웨어 가속 레이-삼각형 교차 테스트

```cpp
// 전통적 소프트웨어 방식 (느림)
bool ray_triangle_intersect_software(Ray ray, Triangle tri) {
    // Möller-Trumbore 알고리즘
    vec3 edge1 = tri.v1 - tri.v0;
    vec3 edge2 = tri.v2 - tri.v0;
    vec3 h = cross(ray.direction, edge2);
    float a = dot(edge1, h);

    // ... 20+ 연산

    return t > 0 && u >= 0 && v >= 0 && u + v <= 1;
}

// RT Core 하드웨어 가속 (10배 빠름)
__device__ float intersect_triangle_rtcore(
    Ray ray,
    Triangle tri,
    float* t_out
) {
    // 하드웨어에서 직접 처리!
    // 단일 클럭 사이클에 완료
    return __rt_trace_triangle(ray, tri, t_out);
}
```

**성능**:
- 초당 10 Giga Rays 처리
- BVH (Bounding Volume Hierarchy) 순회 가속
- 소프트웨어 대비 10-15배 빠름

### 2. Tensor Core (AI 가속)

**기능**: 딥러닝 기반 노이즈 제거 (Denoising)

```python
# 문제: Path Tracing은 노이즈가 많음
samples_for_clean_image = 1000  # 매우 느림

# 해결: Tensor Core + AI Denoiser
samples_with_ai_denoiser = 32   # 30배 빠름!
denoised_image = optix_ai_denoiser(noisy_image, albedo, normal)
```

**OptiX AI Denoiser 사용 예시**:

```cpp
#include <optix_denoiser.h>

void denoise_render(float3* noisy_image, int width, int height) {
    // OptiX Denoiser 초기화
    OptixDenoiser denoiser;
    OptixDenoiserOptions options = {};
    options.guideAlbedo = 1;
    options.guideNormal = 1;

    optixDenoiserCreate(context, &options, &denoiser);

    // Tensor Core에서 실행
    optixDenoiserInvoke(
        denoiser,
        stream,
        &params,
        denoiserState,
        denoiserScratch,
        &inputLayer,
        1,
        0,
        &outputLayer
    );

    // 결과: 32 샘플이 1000 샘플 품질!
}
```

### 3. CUDA Core

**기능**: 범용 병렬 컴퓨팅

```cpp
// 10,752개 CUDA 코어 활용 예시
__global__ void shade_pixels(
    Pixel* pixels,
    int num_pixels,
    Light* lights,
    Material* materials
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    // 각 픽셀 독립적으로 셰이딩
    // 10,752개 픽셀을 동시에 처리!
    pixels[idx].color = compute_lighting(
        pixels[idx],
        lights,
        materials
    );
}
```

---

## 렌더링 최적화 기법

### 1. BVH (Bounding Volume Hierarchy)

**목적**: 레이-씬 교차 테스트 가속

```
씬 전체 검색 (느림)              BVH 사용 (빠름)
─────────────────              ─────────────
레이마다 모든 삼각형 테스트      레이마다 log(N)번만 테스트

10,000 triangles:
- 브루트포스: 10,000 tests    - BVH: ~13 tests
- 시간: 100ms                  - 시간: 0.13ms (770배 빠름!)
```

**BVH 구조**:

```
                   [Root AABB]
                  /            \
           [Left AABB]      [Right AABB]
           /        \        /         \
      [AABB 1]  [AABB 2] [AABB 3]  [AABB 4]
         |         |        |          |
      Tri 1-5   Tri 6-10 Tri 11-15  Tri 16-20
```

```cpp
struct BVHNode {
    AABB bounds;          // 바운딩 박스
    BVHNode* left;
    BVHNode* right;
    Triangle* triangles;  // 리프 노드만
    int tri_count;
};

__device__ bool intersect_bvh(Ray ray, BVHNode* node, Hit* hit) {
    // 1. 바운딩 박스 테스트 (빠름)
    if (!intersect_aabb(ray, node->bounds)) {
        return false;  // Early exit - 대부분 여기서 종료!
    }

    // 2. 리프 노드면 삼각형 테스트
    if (node->is_leaf()) {
        return intersect_triangles(ray, node->triangles, hit);
    }

    // 3. 재귀적으로 자식 노드 검사
    bool hit_left = intersect_bvh(ray, node->left, hit);
    bool hit_right = intersect_bvh(ray, node->right, hit);

    return hit_left || hit_right;
}
```

### 2. Importance Sampling

**목적**: 중요한 방향에 더 많은 샘플 할당

```python
# Naive 방식 (비효율적)
def naive_monte_carlo(hit_point):
    color = 0
    for i in range(1000):  # 1000개 샘플
        # 반구 전체에서 균등 샘플링
        direction = random_hemisphere_direction()
        color += trace_ray(hit_point, direction)
    return color / 1000

# Importance Sampling (효율적)
def importance_sampling(hit_point, material):
    color = 0
    for i in range(100):  # 100개 샘플만으로 충분!
        # 머티리얼 BRDF에 따라 중요한 방향만 샘플링
        direction = material.sample_brdf(hit_point)
        pdf = material.pdf(direction)
        color += trace_ray(hit_point, direction) / pdf
    return color / 100

# 결과: 10배 적은 샘플로 같은 품질!
```

### 3. Adaptive Sampling

**목적**: 픽셀마다 필요한 만큼만 샘플링

```cpp
__device__ float3 adaptive_sample_pixel(int x, int y, Camera cam) {
    float3 color = 0;
    float variance = INFINITY;
    int samples = 0;

    while (variance > THRESHOLD && samples < MAX_SAMPLES) {
        // 샘플 추가
        Ray ray = cam.generate_ray(x, y);
        color += trace_ray(ray);
        samples++;

        // 분산 계산
        variance = compute_variance(color, samples);
    }

    return color / samples;
}

// 결과:
// - 평평한 영역: 4 samples
// - 복잡한 영역: 256 samples
// - 평균: 50 samples (5배 빠름!)
```

### 4. Tiled Rendering

**목적**: 메모리 효율성 향상

```cpp
// 전체 이미지 한 번에 (메모리 부족 가능)
void render_full_image() {
    // 8K 이미지 = 33M 픽셀 × 16 bytes = 528MB VRAM
    allocate_framebuffer(7680, 4320);  // 메모리 부족!
}

// 타일 단위 렌더링 (메모리 절약)
void render_tiled(int tile_size = 256) {
    for (int ty = 0; ty < height; ty += tile_size) {
        for (int tx = 0; tx < width; tx += tile_size) {
            // 256x256 타일 = 65K 픽셀 × 16 bytes = 1MB만 필요
            render_tile(tx, ty, tile_size);
            save_tile_to_disk(tx, ty);
            free_tile_memory();  // 메모리 재사용
        }
    }
}
```

### 5. Level of Detail (LOD)

**목적**: 거리에 따른 디테일 조절

```cpp
struct LODModel {
    Mesh* high_poly;      // 1M triangles
    Mesh* medium_poly;    // 100K triangles
    Mesh* low_poly;       // 10K triangles
};

__device__ Mesh* select_lod(float distance) {
    if (distance < 10.0f)
        return model.high_poly;
    else if (distance < 50.0f)
        return model.medium_poly;
    else
        return model.low_poly;
}

// 성능 향상:
// 평균 삼각형 수: 1,000,000 → 100,000 (10배 감소)
```

---

## 실전 벤치마크

### Blender Cycles 벤치마크

```python
import bpy
import time

def benchmark_render(device_type):
    # 씬 설정
    scene = bpy.data.scenes["Scene"]
    scene.cycles.samples = 128
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080

    # 디바이스 설정
    scene.cycles.device = device_type

    # 렌더링 시간 측정
    start = time.time()
    bpy.ops.render.render(write_still=True)
    elapsed = time.time() - start

    return elapsed

# 결과
results = {
    "CPU (16-core Xeon)": "342 seconds",
    "RTX 3090 (24GB)": "45 seconds",
    "RTX A6000 (48GB)": "38 seconds",
}

# A6000 장점:
# - 48GB 메모리로 더 큰 씬 가능
# - ECC 메모리로 안정성 향상
# - NVLink로 멀티 GPU 스케일링 우수
```

### V-Ray GPU 벤치마크

```
Scene: "Crown" (고품질 실내)
Resolution: 4K (3840 × 2160)
Quality: Production

CPU Mode (Xeon 32-core):     15m 23s
GPU Mode (RTX A6000 × 1):     2m 18s  (6.7배 빠름)
GPU Mode (RTX A6000 × 2):     1m 12s  (12.8배 빠름, NVLink)
GPU Mode (RTX A6000 × 4):     0m 38s  (24.3배 빠름)
```

---

## 다음 단계

이제 기초를 이해했으니 다음 문서에서는:

1. **RTX A6000 세부 아키텍처** - 하드웨어 스펙과 최적화 방법
2. **렌더 팜 구축** - 여러 GPU를 활용한 분산 렌더링
3. **렌더링 엔진 통합** - Blender, V-Ray, Octane, Redshift 등
4. **클라우드 렌더링 서비스** - GPU 파워 대여 시스템 구축
5. **프로덕션 레벨 구현** - 실제 서비스 운영 방법

---

## 핵심 요약

### CPU vs GPU 렌더링
- CPU: 8-64 코어, 복잡한 로직 처리
- GPU: 10,000+ 코어, 단순 병렬 작업 특화
- **속도 차이: 10-50배**

### RTX 아키텍처 3대 핵심
1. **RT Core**: 레이 트레이싱 하드웨어 가속 (10배 빠름)
2. **Tensor Core**: AI 디노이징 (30배 적은 샘플)
3. **CUDA Core**: 범용 병렬 컴퓨팅 (10,752개)

### RTX A6000의 강점
- 48GB VRAM (큰 씬 처리 가능)
- ECC 메모리 (안정성)
- NVLink (멀티 GPU 확장)
- 전문가용 드라이버 최적화

### 최적화 핵심 기법
1. BVH - 교차 테스트 가속 (770배)
2. Importance Sampling - 효율적 샘플링 (10배)
3. Adaptive Sampling - 픽셀별 최적화 (5배)
4. Tiled Rendering - 메모리 절약
5. LOD - 거리별 디테일 조절

**다음**: RTX A6000 아키텍처 심화 →
