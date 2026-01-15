from constant import *
from stage0_model import Net as Stage0Net
from stage0_common import *
import torch.nn.functional as F

def apply_grayscale_guidance(image_rgb):
    # 1. 
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # 2.h=10
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    
    # 3. CLAHE
    # clipLimit=3.0 أو 4.0 تعطي نتائج ممتازة في إظهار القنوات الضعيفة
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(denoised)
    
    # 4. ResNet
    guidance_img = cv2.cvtColor(contrast_enhanced, cv2.COLOR_GRAY2RGB)
    
    return guidance_img

def to_device(batch, device):
    if isinstance(batch, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    return batch.to(device)

stage0_dir = Path(global_dict["stage0_dir"])
stage0_dir.mkdir(exist_ok=True, parents=True)

print("Loading Stage 0 Model...")
stage0_net = Stage0Net(pretrained=False)
stage0_net = load_net(stage0_net, '/kaggle/input/hengck23-submit-physionet/hengck23-submit-physionet/weight/stage0-last.checkpoint.pth')
stage0_net.to(CUDA0)
stage0_net.eval()

print("Starting Stage 0 Processing...")
for n, sample_id in enumerate(tqdm(valid_id)):
    path = test_dir / f'{sample_id}.png'
    output_path = stage0_dir / f'{sample_id}.png'

    image_original = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_original is None: 
        continue
    image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
    
    # Grayscale Guidance
    image_for_model = apply_grayscale_guidance(image_original)
    
    # Batch
    batch = image_to_batch(image_for_model)
    batch = to_device(batch, CUDA0) 

    try:
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=FLOAT_TYPE):
            output = stage0_net(batch)
            
        # Keypoints
        # rotated:
        rotated, keypoint = output_to_predict(image_original, batch, output)
        
        # Homography Normalisation
        normalised, _, _ = normalise_by_homography(rotated, keypoint)
        cv2.imwrite(str(output_path), cv2.cvtColor(normalised, cv2.COLOR_RGB2BGR))
        
    except Exception as e:
        # Pipeline
        copyfile(path, output_path)

print(f"Stage 0 finished. Images saved to {stage0_dir}")