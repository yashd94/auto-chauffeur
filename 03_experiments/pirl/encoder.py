class Resnet18_Encoder(nn.Module):
    def __init__(self):
        super(Resnet18_Encoder, self).__init__()
        self.network = resnet18()
        self.network = torch.nn.Sequential(*list(self.network.children())[:-1])        
        self.projection_original_features = nn.Linear(512, 128)       
        
    def forward_once(self, x):
        return self.network(x)   
 
    def return_reduced_image_features(self, original):
        features = self.forward_once(original)
        features = features.view(-1,512)
        features = self.projection_original_features(features)
        return features      
         
    def forward(self, images = None, rotation = None, mode = 0):
        '''
        mode 0: get 128d feature for image,
        mode 1: get 128d feature for image and rotation

        '''
        if mode == 0:
            return self.return_reduced_image_features(images)
        if mode == 1:
            image_features = self.return_reduced_image_features(images)
            rotation_features = self.return_reduced_image_features(rotation)
            return image_features, rotation_features