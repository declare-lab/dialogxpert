from transformers import BertTokenizer, BertModel

import torch
import torch.nn as nn
import torch.nn.functional as F

class QAdapter(nn.Module):

    def __init__(self, dim_size):
        super().__init__()

        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self._plm_model = BertModel.from_pretrained('bert-base-uncased')

        # Freeze all BERT parameters
        for param in self._plm_model.parameters():
            param.requires_grad = False

        self._plm_model.eval()
        
        self._linear1 = nn.Linear(dim_size, 64)
        self._linear2 = nn.Linear(64, 64)
        self._linear3 = nn.Linear(64,1)

    def forward(self, x):

        output = self._linear1(x)
        output = self._linear2(F.relu(output))
        output = self._linear3(F.relu(output))

        return output
    
    def transform_features(self, state_action_pairs):

        # Tokenize
        encoded = self._tokenizer(
            state_action_pairs, 
            return_tensors='pt', 
            truncation=True,
            max_length=512,
            padding=True)

        # Manually truncate to last 512 tokens
        for key in encoded:

            # Only truncate if sequence length exceeds 512
            if encoded[key].shape[1] < 512:
                break
            
            encoded[key] = encoded[key][:, -512:]  # keep last 512 tokens
        
        # Mount to device
        feat_encoded = {
            k: torch.as_tensor(v).cuda()
            for k, v in encoded.items()
        }

        # Convert the output
        feat_output = []

        with torch.no_grad():   
            feat_output = self._plm_model(**feat_encoded, output_hidden_states=True)
            feat_output = feat_output.hidden_states[-1].mean(-2)

        return feat_output
    