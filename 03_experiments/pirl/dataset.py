# Written by Team Crazy Glitch Asians, NYU Deep Learning Spring 2020


class RotationDataset(Dataset):
    def __init__(self, data_dir, scene_index):
        self.data_dir = data_dir
        self.color_transform = torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4)
        self.flips = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomVerticalFlip()]
        self.normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        self.scene_index = scene_index
        
        self.NUM_SAMPLE_PER_SCENE = 126
        self.NUM_IMAGE_PER_SAMPLE = 6
        self.image_names = ['CAM_FRONT_LEFT.jpeg',
                            'CAM_FRONT.jpeg',
                            'CAM_FRONT_RIGHT.jpeg',
                            'CAM_BACK_LEFT.jpeg',
                            'CAM_BACK.jpeg',
                            'CAM_BACK_RIGHT.jpeg']
        
    def __len__(self):
        return self.scene_index.size * self.NUM_SAMPLE_PER_SCENE * self.NUM_IMAGE_PER_SAMPLE
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        scene_id = self.scene_index[index // (self.NUM_SAMPLE_PER_SCENE * self.NUM_IMAGE_PER_SAMPLE)]
        sample_id = (index % (self.NUM_SAMPLE_PER_SCENE * self.NUM_IMAGE_PER_SAMPLE)) // self.NUM_IMAGE_PER_SAMPLE
        image_name = self.image_names[index % self.NUM_IMAGE_PER_SAMPLE]
        image_path = os.path.join(self.data_dir, f'scene_{scene_id}', f'sample_{sample_id}', image_name) 
        original = Image.open(image_path)
        
        image = torchvision.transforms.Resize((320,320))(original)
        image = torchvision.transforms.RandomCrop((256,256))(image)
        
        rotation = torchvision.transforms.Resize((256,256))(image)
        #augmentation - collor jitter
        image = self.color_transform(image)
        rotation = self.color_transform(rotation)
        # augmentation - flips
        image = self.flips[0](image)
        image = self.flips[1](image)
        # augmentation - rotation
        angles = [90,180,270]
        angle = random.choice(angles)
        rotation = torchvision.transforms.functional.rotate(rotation, angle)
        
        # to tensor
        image = torchvision.transforms.functional.to_tensor(image)
        rotation = torchvision.transforms.functional.to_tensor(rotation)
        # normalize 
        image = self.normalize(image)
        rotation = self.normalize(rotation)        
        
        return {'original': image,'rotation': rotation, 'index' : index}