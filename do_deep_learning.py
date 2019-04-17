#now, training phase:
def do_deep_learning(mod,trl,vl,epochs,lr,gpu):
    import torch
    from torch import nn
    from torch import optim
    print_every=5
    steps = 0
    mod.to('cuda')
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(mod.classifier.parameters(), lr )

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs_t, labels_t) in enumerate(trl):
            steps += 1
        
            if torch.cuda.is_available() and gpu=='gpu':
                inputs_t, labels_t = inputs_t.to('cuda'), labels_t.to('cuda')
            optimizer.zero_grad()
            outputs_t = mod.forward(inputs_t)
            loss = criterion(outputs_t, labels_t)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                mod.eval()
                valid_lost = 0
                accuracy=0
            
            
                for ii, (inputs_v,labels_v) in enumerate(vl):
                    optimizer.zero_grad()
                
                    inputs_v, labels_v = inputs_v.to('cuda') , labels_v.to('cuda')
                    mod.to('cuda')
                    with torch.no_grad():    
                        outputs_v = mod.forward(inputs_v)
                        valid_lost = criterion(outputs_v,labels_v)
                        ps = torch.exp(outputs_v).data
                        equality = (labels_v.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
                valid_lost = valid_lost / len(vl)
                accuracy = accuracy /len(vl)
            
            
                print("Epoch: {}/{}  ".format(e+1, epochs),
                      "Training Loss: {:.3f}".format(running_loss/print_every),
                      "Validation Lost: {:.3f}".format(valid_lost),
                      "Accuracy: {:.3f} %".format(accuracy*100))
            
                running_loss = 0