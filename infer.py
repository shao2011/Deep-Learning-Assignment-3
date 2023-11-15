import segmentation_models_pytorch as smp
import gdown, zipfile, torch, os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import pandas as pd
from PIL import Image
import cv2

# 0. Khởi tạo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### 1. Tạo model
model = smp.Unet(encoder_name="resnet34",encoder_weights="imagenet",in_channels = 3, classes = 3)

### 2. Tải file load model
file_id = "1cJJaPqNKttI8lbSuTF7Z70m0uBHe4Gt0"
gdown.download("http://drive.google.com/uc?/export=download&id="+file_id)

### 3. Giải nén
with zipfile.ZipFile("/kaggle/working/Deep-Learning-Assignment-3/save_model.zip", "r") as zip_ref:
    zip_ref.extractall("/kaggle/working")

### 4. Load model
checkpoint = torch.load("/kaggle/working/save_model.pth")
model.load_state_dict(checkpoint["model_state_dict"])

### 5. Infer
class UNetTestDataClass(Dataset):
    def __init__(self,  
                 transform: transforms.Compose,
                 images_path: str = "/kaggle/input/bkai-igh-neopolyp/test/test"):
        super(UNetTestDataClass, self).__init__()
        
        images_list = os.listdir(images_path)
        images_list = [images_path+"/"+i for i in images_list]
        
        self.images_list = images_list
        self.transform = transform
        
    def __getitem__(self, index):
        img_path = self.images_list[index]
        data = Image.open(img_path)
        h = data.size[1]
        w = data.size[0]
        data = self.transform(data) / 255        
        return data, img_path, h, w
    
    def __len__(self):
        return len(self.images_list)

test_dataset = UNetTestDataClass(transforms.Compose([transforms.Resize((800, 1280))], transforms.PILToTensor()))
test_dataloader = DataLoader(test_dataset,
                         batch_size = 8
                        )

model.eval()
if not os.path.isdir("/kaggle/working/predicted_masks"):
    os.mkdir("/kaggle/working/predicted_masks")
for _, (img, path, H, W) in enumerate(test_dataloader):
    a = path
    b = img
    h = H
    w = W
    
    with torch.no_grad():
        predicted_mask = model(b.to(device))
    for i in range(len(a)):
        image_id = a[i].split('/')[-1].split('.')[0]
        filename = image_id + ".png"
        mask2img = transforms.Resize((h[i].item(), w[i].item()), interpolation=InterpolationMode.NEAREST)(transforms.ToPILImage()(torch.nn.functional.one_hot(torch.argmax(predicted_mask[i], 0)).permute(2, 0, 1).float()))
        mask2img.save(os.path.join("/kaggle/working/predicted_masks/", filename))
    

def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

def rle_encode_one_mask(mask):
    pixels = mask.flatten()
    pixels[pixels > 0] = 255
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle_to_string(rle)

def mask2string(dir):
    ## mask --> string
    strings = []
    ids = []
    ws, hs = [[] for i in range(2)]
    for image_id in os.listdir(dir):
        id = image_id.split('.')[0]
        path = os.path.join(dir, image_id)
        print(path)
        img = cv2.imread(path)[:,:,::-1]
        h, w = img.shape[0], img.shape[1]
        for channel in range(2):
            ws.append(w)
            hs.append(h)
            ids.append(f'{id}_{channel}')
            string = rle_encode_one_mask(img[:,:,channel])
            strings.append(string)
    r = {
        'ids': ids,
        'strings': strings,
    }
    return r


MASK_DIR_PATH = '/kaggle/working/predicted_masks' # change this to the path to your output mask folder
dir = MASK_DIR_PATH
res = mask2string(dir)
df = pd.DataFrame(columns=['Id', 'Expected'])
df['Id'] = res['ids']
df['Expected'] = res['strings']
df.to_csv(r'output.csv', index=False)
