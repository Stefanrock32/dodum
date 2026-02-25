"""
ComfyUI Wan2.2 Venom Video Generation - Modal.com Deployment

This project deploys ComfyUI with Wan2.2 Venom video generation capabilities on Modal.com.
It provides a cloud-based GPU-accelerated environment for running AI video generation workflows.
"""
import os
import sys
import shutil
import time
import subprocess
import modal
from pathlib import Path

MODELS_BASE = "/cache/models"
INPUT_DIR = "/cache/input"
OUTPUT_DIR = "/cache/output"
TORCH_CACHE = "/cache/torch_compile"
CUSTOM_NODES_DIR = "/workspace/ComfyUI/custom_nodes"

MODEL_DIRS = {
    "unet": f"{MODELS_BASE}/diffusion_models",
    "clip": f"{MODELS_BASE}/clip",
    "clip_vision": f"{MODELS_BASE}/clip_vision",
    "vae": f"{MODELS_BASE}/vae",
    "loras": f"{MODELS_BASE}/loras",
    "detection": f"{MODELS_BASE}/detection",
    "ultralytics_bbox": f"{MODELS_BASE}/ultralytics/bbox",
    "sams": f"{MODELS_BASE}/sams",
    "onnx": f"{MODELS_BASE}/onnx",
    "wav2lip": f"{MODELS_BASE}/wav2lip",
}

app = modal.App("comfyui-wan22-venom-master")
volume = modal.Volume.from_name("comfy-storage", create_if_missing=True)


def _patch_wanwrapper(nodes_py: Path) -> int:
    if not nodes_py.exists():
        print(f"[WARN] nodes.py not found: {nodes_py}")
        return 0
    content = nodes_py.read_text(encoding="utf-8")
    original = content
    import re
    patterns = [
        (r'\bH\b,\s*\bH\b', 'H, W'),
        (r'\bh\b,\s*\bh\b', 'h, w'),
        (r'_h,\s*_h', '_h, _w'),
        (r'_H,\s*_H', '_H, _W'),
    ]
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    if content != original:
        nodes_py.write_text(content, encoding="utf-8")
        count = sum(1 for p, r in patterns if p in original)
        print(f"[PATCH] WanVideoWrapper: patched {count} occurrences")
        return count
    else:
        print("[PATCH] WanVideoWrapper: no patch needed")
        return 0


def safe_download(repo: str, filepath: str, out_dir: str, target_name: str, 
                  token: str, max_retries: int = 3) -> None:
    os.makedirs(out_dir, exist_ok=True)
    target_path = os.path.join(out_dir, target_name)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 10 * 1024 * 1024:
        print(f"[SKIP] {target_name} already cached")
        return
    from huggingface_hub import hf_hub_download
    for attempt in range(max_retries):
        try:
            print(f"[DOWNLOAD] {target_name} (attempt {attempt + 1}/{max_retries})")
            downloaded = hf_hub_download(
                repo_id=repo,
                filename=filepath,
                local_dir=out_dir,
                token=token,
                resume_download=True
            )
            if downloaded != target_path:
                shutil.move(downloaded, target_path)
            print(f"[DONE] {target_name}")
            break
        except Exception as e:
            print(f"[ERROR] Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"[RETRY] Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"[FAIL] Failed to download {target_name} after {max_retries} attempts")


def download_file(url: str, out_path: str, max_retries: int = 3) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 10 * 1024 * 1024:
        print(f"[SKIP] {os.path.basename(out_path)} already cached")
        return
    import requests
    for attempt in range(max_retries):
        try:
            print(f"[DOWNLOAD] {os.path.basename(out_path)} (attempt {attempt + 1}/{max_retries})")
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            with open(out_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"[DONE] {os.path.basename(out_path)}")
            break
        except Exception as e:
            print(f"[ERROR] Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"[RETRY] Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"[FAIL] Failed to download {os.path.basename(out_path)} after {max_retries} attempts")


comfy_image = (
    modal.Image.debian_slim(python_version='3.11')
    .apt_install(
        "git", "wget", "curl", "ffmpeg", "libgl1", "libglib2.0-0", 
        "build-essential", "cmake", "python3-dev", "libsndfile1", "pkg-config"
    )
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "torchaudio==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "huggingface_hub",
        "requests",
        "tqdm",
        "transformers",
        "accelerate",
        "sentencepiece",
        "einops",
        "onnxruntime-gpu",
        "ultralytics",
        "diffusers>=0.29.0",
        "pygit2",
        "sageattention",
        "gguf",
        "scipy",
    )
    .pip_install("opencv-python-headless==4.10.0.84")
    .pip_install("librosa")
    .run_commands("mkdir -p /workspace")
    .run_commands("cd /workspace && git clone https://github.com/comfyanonymous/ComfyUI.git")
    .run_commands("cd /workspace/ComfyUI && pip install -r requirements.txt")
    .run_commands("pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python || true")
    .run_commands("pip install --no-cache-dir opencv-python-headless==4.10.0.84")
    .run_commands("cd /workspace/ComfyUI/custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Manager.git")
    .run_commands("cd /workspace/ComfyUI/custom_nodes && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git")
    .run_commands("cd /workspace/ComfyUI/custom_nodes && git clone https://github.com/kijai/ComfyUI-KJNodes.git")
    .run_commands("cd /workspace/ComfyUI/custom_nodes && git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git")
    .run_commands("cd /workspace/ComfyUI/custom_nodes && git clone https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch.git")
    .run_commands("cd /workspace/ComfyUI/custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git")
    .run_commands("cd /workspace/ComfyUI/custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git")
    .run_commands("cd /workspace/ComfyUI/custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git")
    .run_commands("cd /workspace/ComfyUI/custom_nodes/ComfyUI-Impact-Pack && python install.py")
    .run_commands("cd /workspace/ComfyUI/custom_nodes/ComfyUI-Impact-Subpack && pip install -r requirements.txt || true")
    .run_commands("cd /workspace/ComfyUI/custom_nodes && git clone https://github.com/ShmuelRonen/ComfyUI_wav2lip.git")
    .run_commands("cd /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite && pip install -r requirements.txt || true")
    .run_commands("cd /workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes && pip install -r requirements.txt || true")
    .run_commands("cd /workspace/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper && pip install -r requirements.txt || true")
    .run_commands("cd /workspace/ComfyUI/custom_nodes/ComfyUI-Inpaint-CropAndStitch && pip install -r requirements.txt || true")
    .run_commands("cd /workspace/ComfyUI/custom_nodes/ComfyUI-Impact-Pack && python install.py")
    .run_commands("cd /workspace/ComfyUI/custom_nodes/ComfyUI_wav2lip && pip install -r requirements.txt || true")
)


@app.function(
    image=comfy_image,
    volumes={'/cache': volume},
    secrets=[modal.Secret.from_name('huggingface-secret')],
    timeout=14400,
    cpu=4.0,
    memory=8192
)
def download_models():
    volume.reload()
    token = os.environ.get('HF_TOKEN')
    DOWNLOADS = [
        ('Comfy-Org/Wan_2.1_ComfyUI_repackaged',
         'split_files/clip_vision/clip_vision_h.safetensors',
         MODEL_DIRS['clip_vision'], 'clip_vision_h.safetensors'),
        ('Comfy-Org/Wan_2.2_ComfyUI_Repackaged',
         'split_files/vae/wan_2.1_vae.safetensors',
         MODEL_DIRS['vae'], 'wan_2.1_vae.safetensors'),
        ('Comfy-Org/Wan_2.1_ComfyUI_repackaged',
         'split_files/text_encoders/umt5_xxl_fp16.safetensors',
         MODEL_DIRS['clip'], 'umt5_xxl_fp16.safetensors'),
        ('Comfy-Org/Wan_2.2_ComfyUI_Repackaged',
         'split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors',
         MODEL_DIRS['unet'], 'wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors'),
        ('Bingsu/adetailer',
         'face_yolov8m.pt',
         MODEL_DIRS['ultralytics_bbox'], 'face_yolov8m.pt'),
    ]
    for repo, filepath, out_dir, name in DOWNLOADS:
        safe_download(repo, filepath, out_dir, name, token)
    
    download_file(
        'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/wav2lip.pth',
        os.path.join(MODEL_DIRS['wav2lip'], 'wav2lip.pth')
    )
    download_file(
        'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/face_detection_yolov8n.onnx',
        os.path.join(MODEL_DIRS['wav2lip'], 'face_detection_yolov8n.onnx')
    )
    
    volume.commit()
    print("[DONE] All models downloaded!")


# NOTE: max_containers=1 - This function MUST run with only 1 container (required for ComfyUI web server statefulness)
@app.function(
    image=comfy_image,
    gpu='A100-80GB',
    timeout=86400,
    volumes={'/cache': volume},
    memory=65536,
    max_containers=1
)
@modal.web_server(port=8188, startup_timeout=900)
@modal.concurrent(max_inputs=100)
def serve():
    volume.reload()
    os.environ['TORCH_COMPILE_CACHE_DIR'] = TORCH_CACHE
    for d in list(MODEL_DIRS.values()) + [INPUT_DIR, OUTPUT_DIR, TORCH_CACHE]:
        os.makedirs(d, exist_ok=True)
    os.makedirs(MODEL_DIRS['wav2lip'], exist_ok=True)
    nodes_py = Path(f'{CUSTOM_NODES_DIR}/ComfyUI-WanVideoWrapper/nodes.py')
    _patch_wanwrapper(nodes_py)
    comfy_models = "/workspace/ComfyUI/models"
    symlink_map = {
        "detection": MODEL_DIRS['detection'],
        "ultralytics": MODEL_DIRS['ultralytics_bbox'],
        "sams": MODEL_DIRS['sams'],
        "onnx": MODEL_DIRS['onnx'],
        "wav2lip": MODEL_DIRS['wav2lip'],
    }
    for name, target in symlink_map.items():
        link_path = os.path.join(comfy_models, name)
        if os.path.islink(link_path):
            os.unlink(link_path)
        elif os.path.exists(link_path):
            shutil.rmtree(link_path)
        os.symlink(target, link_path)
        print(f"[SYMLINK] {link_path} -> {target}")
    zzz_paths_fix = '''"""
Path fix for detection/sams/ultralytics/onnx/wav2lip models.
"""
from folder_paths import folder_paths
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for name, path in {
    "detection": "/cache/models/detection",
    "sams": "/cache/models/sams",
    "ultralytics": "/cache/models/ultralytics/bbox",
    "onnx": "/cache/models/onnx",
    "wav2lip": "/cache/models/wav2lip",
}.items():
    if name not in folder_paths.folder_names:
        folder_paths.folder_names[name] = []
    if path not in folder_paths.folder_names[name]:
        folder_paths.folder_names[name].append(path)
'''
    zzz_path = os.path.join(CUSTOM_NODES_DIR, 'zzz_paths_fix.py')
    Path(zzz_path).write_text(zzz_paths_fix, encoding="utf-8")
    print(f"[WRITE] {zzz_path}")
    extra_paths_yaml = '''modal_storage:
  base_path: /cache/models
  unet: diffusion_models
  clip: clip
  vae: vae
  loras: loras
  clip_vision: clip_vision
  controlnet: controlnet
  upscalers: upscalers
  embeddings: embeddings
  vae_approx: vae_approx
  wav2lip: wav2lip
'''
    yaml_path = "/workspace/ComfyUI/extra_model_paths.yaml"
    Path(yaml_path).write_text(extra_paths_yaml, encoding="utf-8")
    print(f"[WRITE] {yaml_path}")
    os.chdir('/workspace/ComfyUI')
    return subprocess.Popen([
        sys.executable, 'main.py',
        '--listen', '0.0.0.0',
        '--port', '8188',
        '--input-directory', INPUT_DIR,
        '--output-directory', OUTPUT_DIR
    ])


@app.local_entrypoint()
def main():
    download_models.remote()
    print("Done! Run: modal deploy app.py")


if __name__ == "__main__":
    pass
