import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader


class FloderData():
    def __init__(self, batch_size, train_path, valid_path, test_path, num_w, load_model):
        self.bs = batch_size
        self.train_path = train_path
        self.vaild_path = valid_path
        self.test_path = test_path
        self.num_w = num_w
        self.load_model = load_model
        self.image_transforms = {
            # Train uses data augmentation
            'train':
                transforms.Compose([
                    # transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(),
                    transforms.RandomHorizontalFlip(),
                    # transforms.CenterCrop(size=224),  # Image net standards
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])  # Imagenet standards
                ]),
            # Validation does not use augmentation
            'valid':
                transforms.Compose([
                    # transforms.Resize(size=256),
                    # transforms.CenterCrop(size=224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            'test':
                transforms.Compose([
                    # transforms.Resize(size=224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
        }

    def get_dataloader(self):
        if self.load_model == 'train':
            # Datasets from folders
            data = {
                'train':
                    datasets.ImageFolder(root=self.train_path, transform=self.image_transforms['train']),
                'valid':
                    datasets.ImageFolder(root=self.vaild_path, transform=self.image_transforms['valid']),
            }

            # Dataloader iterators, make sure to shuffle
            dataloaders = {
                'train': DataLoader(data['train'], batch_size=self.bs, num_workers=self.num_w, shuffle=True),
                'valid': DataLoader(data['valid'], batch_size=self.bs, num_workers=self.num_w, shuffle=False)
            }
        elif self.load_model == 'test':
            # Datasets from folders
            data = {
                'test':
                    datasets.ImageFolder(root=self.test_path, transform=self.image_transforms['test']),
            }

            # Dataloader iterators, make sure to shuffle
            dataloaders = {
                'test': DataLoader(data['test'], batch_size=self.bs, num_workers=self.num_w, shuffle=False),
            }
        else:
            print('model type choose error')
        return dataloaders
