# A class file that contains the configuration for the application

import os
import torch
from torchvision import transforms

# Define the configuration class

class Config:
    def __init__(self, mode='train'):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5])
        ])

        
        # Model parameters
        self.num_layers =  3 # Default 6
        self.num_heads = 4   # Default 8 
        self.seq_dim = 43 # 43개의 토큰 
        self.seq_embedding_dim = 64 # 
        self.param_dim = 5 + 4 + 3 + 4 # 5 for shoot, 4 for the internode, 3 for the petiole, 4 for the leaf
        self.param_embedding_dim = 64
        
        # self.data_dir = 'data'
        # self.model_path = 'model.pth'
        # self.num_classes = 4
        # self.batch_size = 32
        # self.epochs = 10
        # self.lr = 0.001
        # self.img_size = 224
        # self.transforms = {
        #     'train': {
        #         'resize': (256, 256),
        #         'mean': (0.485, 0.456, 0.406),
        #         'std': (0.229, 0.224, 0.225)
        #     },
        #     'val': {
        #         'resize': (256, 256),
        #         'mean': (0.485, 0.456, 0.406),
        #         'std': (0.229, 0.224, 0.225)
        #     }
        # }
        # self.model = nn.Sequential(
        #     nn.Conv2d(3, 16, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(16, 32, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(32, 64, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 128, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Flatten(),
        #     nn.Linear(128*14*14, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, self.num_classes)
        # )
        # self.loss_fn = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)