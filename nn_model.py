def nn_model(model_name,hidden_layer_in,hidden_layer_out, dropout):
    import torchvision.models as models
    from torch import nn
    hli=hidden_layer_in
    hlo=hidden_layer_out
    p=dropout
    lookup={"vgg16":25088,"alexnet":9216}
    if model_name not in lookup:
        raise Exception('Error! choose alexnet or vgg16 only!') 
    else:
        model=getattr(models,model_name)(pretrained=True)
        fea_in=lookup.get(model_name)
        for param in model.parameters():
            param.requires_grad = False
            
        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
        ('dropout',nn.Dropout(p)),
        ('fc1', nn.Linear(fea_in, hli)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hli, hlo)),
        ('relu', nn.ReLU()),
        ('fc3', nn.Linear(hlo, 102)),
        ('output', nn.LogSoftmax(dim=1))]))
    
    model.classifier = classifier
    return model