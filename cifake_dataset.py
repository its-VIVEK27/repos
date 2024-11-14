from os import listdir, walk
from os.path import isdir, join
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class CIFAKEDataset(Dataset):
    def __init__(self, dataset_path, resolution=224, norm_mean=IMAGENET_DEFAULT_MEAN, norm_std=IMAGENET_DEFAULT_STD):
        assert isdir(dataset_path), f"got {dataset_path}"
        self.dataset_path = dataset_path
        
        # Parse images in REAL and FAKE folders
        self.items = self.parse_dataset()
        
        # Sets up the preprocessing options
        assert isinstance(resolution, int) and resolution >= 1, f"got {resolution}"
        self.resolution = resolution
        assert len(norm_mean) == 3
        self.norm_mean = norm_mean
        assert len(norm_std) == 3
        self.norm_std = norm_std

    def parse_dataset(self):
        def is_image(filename):
            for extension in ["jpg", "png", "jpeg"]:
                if filename.lower().endswith(extension):
                    return True
            return False

        real_path = join(self.dataset_path, "REAL")
        fake_path = join(self.dataset_path, "FAKE")

        # Collect images from REAL and FAKE folders, including subdirectories
        items = []
        for label, folder in [("REAL", True), ("FAKE", False)]:
            folder_path = join(self.dataset_path, label)
            for root, _, files in walk(folder_path):
                for file in files:
                    if is_image(file):
                        items.append({
                            "image_path": join(root, file),
                            "is_real": folder
                        })
        return items

    def __len__(self):
        return len(self.items)

    def read_image(self, path):
        image = Image.open(path).convert('RGB')
        image = T.Compose([
            T.Resize(self.resolution + self.resolution // 8, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(self.resolution),
            T.ToTensor(),
            T.Normalize(mean=self.norm_mean, std=self.norm_std),
        ])(image)
        return image
    
    def __getitem__(self, i):
        sample = {
            "image_path": self.items[i]["image_path"],
            "image": self.read_image(self.items[i]["image_path"]),
            "is_real": torch.as_tensor([1 if self.items[i]["is_real"] is True else 0]),
        }
        return sample
