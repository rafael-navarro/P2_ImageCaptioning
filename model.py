import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.batchNorm = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.batchNorm(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        
        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        # the LSTM takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)

        # the linear layer that maps the hidden state output dimension 
        
        self.hidden2word = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        
        embeds = self.word_embeddings(captions)
        embeds = torch.cat((features.unsqueeze(1),embeds[:, :-1,:]), dim=1)
        
        lstm_out, self.hidden = self.lstm(embeds)
        
        outputs = self.hidden2word(lstm_out)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        #print(inputs)
        
        hidden_state = None
        output = []
        for _ in range(max_len):
            lstm_out, hidden_state = self.lstm(inputs, hidden_state) 
            outputs = self.hidden2word(lstm_out)  
            pred=torch.argmax(outputs,dim=2)
            
            output.append(pred.item()) 

            inputs = self.word_embeddings(pred) 
            
        return output
