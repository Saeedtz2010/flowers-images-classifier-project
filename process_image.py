def process_image(image):
    from PIL import Image
    from torchvision import datasets, transforms, models
    
    I = Image.open(image)

    transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])
    
    return transform(I)