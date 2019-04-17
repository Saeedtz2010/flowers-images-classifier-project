def load_model(path):
    import torch
    from nn_model import nn_model
    checkpoint = torch.load(path)
    model_name = checkpoint['model_name']
    hidden_layer_in = checkpoint['hidden_layer_in']
    hidden_layer_out = checkpoint['hidden_layer_out']
    dropout=checkpoint['dropout']
    model=nn_model(model_name,hidden_layer_in,hidden_layer_out, dropout)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model