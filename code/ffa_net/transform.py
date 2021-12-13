import torch
import torchvision
from torch import nn, Tensor
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T

mean = [0.64,0.6,0.58],
std = [0.14,0.15,0.152]

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class CropSameSize(nn.Module):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        
        image_size = image.size[::-1]
        target = T.CenterCrop(image_size)(target)
        return image, target

class ToTensor(nn.Module):
    def forward(
                self, 
                image: Tensor, 
                target: Tensor
                ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.pil_to_tensor(image)
        image = F.convert_image_dtype(image)
        
        target = F.pil_to_tensor(target)
        target = F.convert_image_dtype(target)
        
        return image, target
    
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
        
    def forward(self, image, target):
        image = T.Normalize(self.mean, self.std)(image)
        target = T.Normalize(self.mean, self.std)(target)
        return image, target


def get_transform(train):
    transforms = []
    if train:        
        transforms.append(CropSameSize())
    transforms.append(T.Resize(460))
    transforms.append(ToTensor())
    transforms.append(Normalize(mean, std))
    
    return Compose(transforms)