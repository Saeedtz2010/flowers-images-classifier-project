def pred(image_path, model,gpu):
    from process_image import process_image
    import torch
    import torch.nn.functional as F

    if torch.cuda.is_available() and gpu=='gpu':
        model.to('cuda')
    img = process_image(image_path)
    img = img.unsqueeze_(0)
    
    with torch.no_grad():
        output = model.forward(img.to('cuda'))
        
    p = F.softmax(output.data,dim=1)
    probs, classes = p.topk(5)
    return probs[0], classes[0]