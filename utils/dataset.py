# # utils/dataset.py
# from torch.utils.data import Dataset
# from PIL import Image
# import os
# import torchvision.transforms as transforms
# import cv2
# import numpy as np
# from utils.preprocessing import ImagePreprocessor

# class FiberglassDataset(Dataset):
#     def __init__(self, root_dir, is_train=True, preprocessing=False, transform=None):
#         """
#         Args:
#             root_dir (str): Root directory of the dataset
#             is_train (bool): Whether this is training or test data
#             preprocessing (bool): Whether to apply preprocessing enhancements
#             transform: Optional transform to be applied to the images
#         """
#         self.root_dir = root_dir
#         self.is_train = is_train
#         self.preprocessor = ImagePreprocessor() if preprocessing else None
        
#         # Default transform if none provided
#         self.transform = transform or transforms.Compose([
#             transforms.Resize((256, 256)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                std=[0.229, 0.224, 0.225])
#         ])
        
#         # Setup file paths
#         if is_train:
#             self.image_dir = os.path.join(root_dir, 'train', 'normal')
#             self.image_files = [f for f in os.listdir(self.image_dir)
#                               if f.endswith(('.jpg', '.png', '.jpeg'))]
#         else:
#             self.image_dir = os.path.join(root_dir, 'test')
#             self.normal_files = [os.path.join('normal', f) 
#                                for f in os.listdir(os.path.join(self.image_dir, 'normal'))
#                                if f.endswith(('.jpg', '.png', '.jpeg'))]
#             self.anomaly_files = [os.path.join('anomalous', f)
#                                 for f in os.listdir(os.path.join(self.image_dir, 'anomalous'))
#                                 if f.endswith(('.jpg', '.png', '.jpeg'))]
#             self.image_files = self.normal_files + self.anomaly_files
#             self.labels = [0] * len(self.normal_files) + [1] * len(self.anomaly_files)

#     def __len__(self):
#         return len(self.image_files)
    
#     def __getitem__(self, idx):
#         if self.is_train:
#             img_path = os.path.join(self.image_dir, self.image_files[idx])
#         else:
#             img_path = os.path.join(self.image_dir, self.image_files[idx])
            
#         # Load and convert image
#         image = Image.open(img_path).convert('RGB')
        
#         # Apply preprocessing if enabled
#         if self.preprocessor:
#             image = np.array(image)
#             image = self.preprocessor.enhance_defects(image)
#             image = Image.fromarray((image * 255).astype(np.uint8))
        
#         # Apply transforms
#         if self.transform:
#             image = self.transform(image)
            
#         if self.is_train:
#             return image
#         else:
#             return image, self.labels[idx]

# utils/dataset.py
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms
import cv2
import numpy as np
from utils.preprocessing import ImagePreprocessor

class FiberglassDataset(Dataset):
    def __init__(self, root_dir, is_train=True, preprocessing=False, transform=None):
        """
        Args:
            root_dir (str): Root directory of the dataset
            is_train (bool): Whether this is training or test data
            preprocessing (bool): Whether to apply preprocessing enhancements
            transform: Optional transform to be applied to the images
        """
        self.root_dir = root_dir
        self.is_train = is_train
        self.preprocessor = ImagePreprocessor() if preprocessing else None
        
        # Default transforms with augmentations for training
        if transform is None:
            if is_train:
                self.transform = transforms.Compose([
                    #transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
                    transforms.RandomResizedCrop(256, scale=(0.9, 1.0)),  # Less aggressive crop
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    #transforms.RandomRotation(90),
                    transforms.RandomRotation(15), # redce rotation
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    #transforms.ColorJitter(brightness=0.05, contrast=0.05), #reduce intenisty 
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
        
        # Setup file paths
        if is_train:
            self.image_dir = os.path.join(root_dir, 'train', 'normal')
            self.image_files = [f for f in os.listdir(self.image_dir)
                              if f.endswith(('.jpg', '.png', '.jpeg'))]
        else:
            self.image_dir = os.path.join(root_dir, 'test')
            self.normal_files = [os.path.join('normal', f) 
                               for f in os.listdir(os.path.join(self.image_dir, 'normal'))
                               if f.endswith(('.jpg', '.png', '.jpeg'))]
            self.anomaly_files = [os.path.join('anomalous', f)
                                for f in os.listdir(os.path.join(self.image_dir, 'anomalous'))
                                if f.endswith(('.jpg', '.png', '.jpeg'))]
            self.image_files = self.normal_files + self.anomaly_files
            self.labels = [0] * len(self.normal_files) + [1] * len(self.anomaly_files)

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        if self.is_train:
            img_path = os.path.join(self.image_dir, self.image_files[idx])
        else:
            img_path = os.path.join(self.image_dir, self.image_files[idx])
            
        # Load and convert image
        image = Image.open(img_path).convert('RGB')
        
        # Apply preprocessing if enabled
        if self.preprocessor:
            image = np.array(image)
            image = self.preprocessor.enhance_defects(image)
            image = Image.fromarray((image * 255).astype(np.uint8))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        if self.is_train:
            return image
        else:
            return image, self.labels[idx]