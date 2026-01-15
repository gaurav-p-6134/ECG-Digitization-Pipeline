from constant import *
from stage1_model import Net as Stage1Net
from stage1_common import *

stage0_dir = Path(global_dict["stage0_dir"])
stage1_dir = Path(global_dict["stage1_dir"])
stage1_dir.mkdir(exist_ok=True, parents=True)

stage1_net = Stage1Net(pretrained=False)
stage1_net = load_net(stage1_net, '/kaggle/input/hengck23-submit-physionet/hengck23-submit-physionet/weight/stage1-last.checkpoint.pth')
stage1_net.to(CUDA0)
stage1_net.eval()

for n, sample_id in enumerate(tqdm(valid_id)):
    path = stage0_dir / f'{sample_id}.png'
    output_path = stage1_dir / f'{sample_id}.png'
    
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None: 
        image = cv2.imread(str(test_dir / f'{sample_id}.png'), cv2.IMREAD_COLOR)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    batch = {'image': torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0).to(CUDA0)}

    try:
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=FLOAT_TYPE):
            output = stage1_net(batch)
        gridpoint_xy, _ = output_to_predict(image, batch, output)
        rectified = rectify_image(image, gridpoint_xy)
        cv2.imwrite(str(output_path), cv2.cvtColor(rectified, cv2.COLOR_RGB2BGR))
    except Exception as e:
        copyfile(path, output_path)