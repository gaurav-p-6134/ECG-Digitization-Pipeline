import sys
sys.path.append('/kaggle/input/hengck23-submit-physionet/hengck23-submit-physionet')
from constant import *
import torchvision.transforms as T
from stage2_model import *
from stage2_common import *
from scipy.signal import savgol_filter
import timm
import traceback
from pathlib import Path
import os
import torch.nn as nn
import cv2

# --- AUTO-FIND WEIGHTS ---
def find_weights(filename="iter_0004200.pt"):
    for p in Path("/kaggle/input").rglob(filename): return str(p)
    raise FileNotFoundError(f"Missing {filename}")

# --- MODEL ---
class Net3(nn.Module):
    def __init__(self, pretrained=True):
        super(Net3, self).__init__()
        encoder_dim = [64, 128, 256, 512]
        decoder_dim = [128, 64, 32, 16]
        self.encoder = timm.create_model('resnet34.a3_in1k', pretrained=pretrained, in_chans=3, num_classes=0, global_pool='')
        self.decoder = MyCoordUnetDecoder(in_channel=encoder_dim[-1], skip_channel=encoder_dim[:-1][::-1] + [0], out_channel=decoder_dim, scale=[2, 2, 2, 2])
        self.pixel = nn.Conv2d(decoder_dim[-1], 4, 1)

    def forward(self, image):
        encode = encode_with_resnet(self.encoder, image)
        last, _ = self.decoder(feature=encode[-1], skip=encode[:-1][::-1] + [None])
        pixel = self.pixel(last)
        return pixel

# --- IMAGE ENHANCEMENT HELPERS ---
def apply_sharpen(img_norm):
    # Un-normalize to uint8 for CV2 ops
    img_uint8 = (img_norm * 255).astype(np.uint8)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) # Moderate Sharpen
    sharp = cv2.filter2D(img_uint8, -1, kernel)
    return sharp.astype(np.float32) / 255.0

def apply_clahe(img_norm):
    img_uint8 = (img_norm * 255).astype(np.uint8)
    # Convert to LAB to apply CLAHE only on Lightness channel
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final.astype(np.float32) / 255.0

# --- SAFE SIGNAL HELPERS (Rank 1 Logic) ---
def get_baseline_mode(signal, bins=2048):
    try:
        hist, bin_edges = np.histogram(signal, bins=bins)
        max_idx = np.argmax(hist)
        return (bin_edges[max_idx] + bin_edges[max_idx+1]) / 2.0
    except: return np.median(signal)

def safe_savgol(y, window=5, poly=3):
    try:
        pad_len = window // 2 + 5
        y_pad = np.pad(y, (pad_len, pad_len), mode='reflect')
        y_smooth = savgol_filter(y_pad, window, poly)
        return y_smooth[pad_len:-pad_len]
    except: return y

# --- CONFIG ---
VOLTAGE_RESOLUTION = 78.74
TIME_START, TIME_END = 235, 4161
ZERO_LEVELS = [703.5, 987.5, 1271.5, 1531.5]

stage1_dir = Path(global_dict["stage1_dir"])
stage2_dir = Path(global_dict["stage2_dir"])
stage2_dir.mkdir(exist_ok=True, parents=True)

stage2_net = Net3(pretrained=False).to(CUDA0)
stage2_net.load_state_dict(torch.load(find_weights("iter_0004200.pt"), map_location=CUDA0))

if torch.cuda.device_count() > 1:
    print(f"🔥 TITANIUM MODE: Using {torch.cuda.device_count()} GPUs")
    stage2_net = nn.DataParallel(stage2_net)

stage2_net.eval()
resize = T.Resize((1696, 4352), interpolation=T.InterpolationMode.BICUBIC)

print("Starting Enhanced Extraction...")
for n, sample_id in enumerate(tqdm(valid_id)):
    path = stage1_dir / f'{sample_id}.png'
    output_path = stage2_dir / f'{sample_id}.npy'
    
    if not path.exists(): continue

    try:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        try: target_len = valid_df[(valid_df['id']==sample_id) & (valid_df['lead']=='II')].iloc[0].number_of_rows
        except: target_len = 5000
        
        crop = image[:1696, :2176]
        crop_norm = crop / 255.0
        
        # --- 5-WAY FEATURE TTA ---
        # 1. Normal
        t_norm = torch.from_numpy(np.ascontiguousarray(crop_norm.transpose(2, 0, 1))).unsqueeze(0)
        # 2. Dark (Noise reduction)
        t_dark = torch.from_numpy(np.ascontiguousarray((crop_norm ** 0.8).transpose(2, 0, 1))).unsqueeze(0)
        # 3. Bright (Grid removal)
        t_bright = torch.from_numpy(np.ascontiguousarray((crop_norm ** 1.2).transpose(2, 0, 1))).unsqueeze(0)
        # 4. Sharpened (Peak definition)
        crop_sharp = apply_sharpen(crop_norm)
        t_sharp = torch.from_numpy(np.ascontiguousarray(crop_sharp.transpose(2, 0, 1))).unsqueeze(0)
        # 5. CLAHE (Weak signal boost)
        crop_clahe = apply_clahe(crop_norm)
        t_clahe = torch.from_numpy(np.ascontiguousarray(crop_clahe.transpose(2, 0, 1))).unsqueeze(0)
        
        batch = torch.cat([t_norm, t_dark, t_bright, t_sharp, t_clahe], dim=0)
        batch = resize(batch).float().to(CUDA0)
        
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=FLOAT_TYPE):
            output = stage2_net(batch)
        
        pixel_maps = torch.sigmoid(output).float().data.cpu().numpy()
        
        # Weighted Ensemble (Favor Sharpness & Normal)
        pixel_avg = (pixel_maps[0] * 0.40) + \
                    (pixel_maps[1] * 0.15) + (pixel_maps[2] * 0.15) + \
                    (pixel_maps[3] * 0.20) + (pixel_maps[4] * 0.10)
        
        # UNSHACKLED EXTRACTION (Highest Fidelity)
        series_in_pixel = pixel_to_series(pixel_avg[..., TIME_START:TIME_END], ZERO_LEVELS, target_len)
        series_mv = (np.array(ZERO_LEVELS).reshape(4, 1) - series_in_pixel) / VOLTAGE_RESOLUTION
        
        for i in range(series_mv.shape[0]):
            s = series_mv[i]
            
            # 1. Mode Centering
            s -= get_baseline_mode(s, bins=2048)
            
            # 2. Safe SavGol (No risky filtering)
            s = safe_savgol(s, window=5, poly=3)
            
            # 3. Adaptive Clip
            p01, p99 = np.percentile(s, 0.1), np.percentile(s, 99.9)
            s = np.clip(s, p01 - 0.5, p99 + 0.5)
            
            series_mv[i] = s
            
        series_mv = np.clip(series_mv, -4.0, 4.0)
        np.save(output_path, series_mv)
        
        if os.path.exists(path): os.remove(path)
        
    except Exception as e:
        traceback.print_exc()
        np.save(output_path, np.zeros((4, 5000)))

print("✅ Stage 2 (Titanium) Complete.")