# 렌더링 엔진 통합 가이드 (Blender, V-Ray, Octane, Redshift)

## Blender Cycles GPU 통합

### Headless 렌더링

```python
# blender_worker.py
import bpy
import sys

def render_job(scene_file, output_path, gpu_ids=[0]):
    # GPU 디바이스 설정
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA'
    
    for i, gpu_id in enumerate(gpu_ids):
        prefs.devices[gpu_id].use = True
    
    # 씬 로드
    bpy.ops.wm.open_mainfile(filepath=scene_file)
    
    # 렌더 설정
    scene = bpy.context.scene
    scene.cycles.device = 'GPU'
    scene.render.filepath = output_path
    
    # 렌더 실행
    bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    scene_file = sys.argv[1]
    output = sys.argv[2]
    gpus = [int(x) for x in sys.argv[3].split(',')]
    
    render_job(scene_file, output, gpus)
```

## V-Ray GPU 통합

```python
# vray_render_node.py
from vray import VRayRenderer

class VRayGPUNode:
    def __init__(self, gpu_ids):
        self.renderer = VRayRenderer()
        self.gpu_ids = gpu_ids
        
    def render(self, scene, output):
        self.renderer.load_scene(scene)
        self.renderer.set_gpus(self.gpu_ids)
        self.renderer.render(output)
```

이어서 AI 학습용 GPU 대여 가이드를 작성하겠습니다...
