import torch
import torch.nn as nn
from torchvision import models

IMAGE_SIZE = 64

class ModelInitializer:
    def __init__(self, model_name, num_classes, resume_from=None, use_pretrained=False):
        self.model_name = model_name
        self.num_classes = num_classes
        self.resume_from = resume_from
        self.use_pretrained = use_pretrained

        self.model_ft = None
        self.input_size = 0

    def initialize(self):
        if self.model_name == "resnet18":
            self._initialize_resnet18()
        elif self.model_name == "resnet50":
            self._initialize_resnet50()
        elif self.model_name == "resnet152":
            self._initialize_resnet152()
        elif self.model_name == "vgg":
            self._initialize_vgg16()
        else:
            raise Exception("Invalid model name!")

        if self.resume_from is not None:
            self._load_weights()

        return self.model_ft, self.input_size

    def _initialize_resnet18(self):
        self.model_ft = models.resnet18(pretrained=self.use_pretrained)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, self.num_classes)
        )
        self.input_size = IMAGE_SIZE

    def _initialize_resnet50(self):
        self.model_ft = models.resnet50(pretrained=self.use_pretrained)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, self.num_classes)
        )
        self.input_size = IMAGE_SIZE

    def _initialize_resnet152(self):
        self.model_ft = models.resnet152(pretrained=self.use_pretrained)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
        self.input_size = IMAGE_SIZE

    def _initialize_vgg16(self):
        self.model_ft = models.vgg16(pretrained=self.use_pretrained)
        num_ftrs = self.model_ft.classifier[6].in_features
        self.model_ft.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
        self.input_size = IMAGE_SIZE

    def _load_weights(self):
        print(f"Loading weights from {self.resume_from}")
        self.model_ft.load_state_dict(torch.load(self.resume_from))


# Sử dụng lớp ModelInitializer
model_initializer = ModelInitializer(
    model_name="resnet18", num_classes=10, resume_from=None)
model, input_size = model_initializer.initialize()
