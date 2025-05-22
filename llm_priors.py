from torch.utils.data import RandomSampler, DataLoader, Dataset
from transformers import default_data_collator
from torch.nn.utils.rnn import pad_sequence

import torch
import datasets

class BufferSet(Dataset):

    def __init__(self, data):
        self._data = data
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        item = self._data[idx]
        return {key: item[key] for key in item}
    
def collate_fn(batch):
    
    keys = batch[0].keys()
    output = {}

    for key in keys:
        
        values = [
            torch.tensor(item[key]) 
            if not isinstance(item[key], torch.Tensor) else item[key]
            for item in batch
        ]

        if isinstance(values[0], torch.Tensor) and values[0].ndim == 2:
            # Pad along dimension 0 (sequence length)
            padded = pad_sequence(values, batch_first=True)
            output[key] = padded
        else:
            
            output[key] = torch.stack(values)
    
    return output

def soft_update(target, source, tau):

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((target_param.data * (1.0 - tau)) + (param.data * tau))

def train_qnetwork(replay_buffer, prob_optimizer, prob_network, target_network, device):

    # Convert to data loader
    train_data = BufferSet(replay_buffer)

    sampler = RandomSampler(train_data, replacement=True, num_samples = 4 * (len(replay_buffer)))

    train_dataloader = DataLoader(train_data, batch_size = 32, sampler = sampler, collate_fn=collate_fn)

    for batch in train_dataloader:

        # Unpack
        current_feat_batch = batch['feat'].float().to(device)
        next_feat_batch = batch['next_feat'].float().to(device)
        reward_batch = batch['reward'].float().to(device)
        done_batch = batch['done'].float().to(device)

        # Get the current (s, a) values
        p = prob_network(current_feat_batch).squeeze(-1).squeeze(-1)

        target_value = []

        # Get the target values (next (s,a) values)
        with torch.no_grad():
            target_p = target_network(next_feat_batch).squeeze(-1)
            greedy_target_p = target_p.max(-1)[0]

            target_value = reward_batch + (0.99 * (1 - done_batch)) * (greedy_target_p)
        
        # Calculate loss
        td_learning_loss = torch.nn.functional.mse_loss(p, target_value.detach(), reduction="sum")
        td_learning_loss.backward()

        # Update gradients
        torch.nn.utils.clip_grad_norm_(prob_network.parameters(), 1.0)
        prob_optimizer.step()
        prob_optimizer.zero_grad()

        # Perform soft update
        soft_update(target_network, prob_network, 0.005)

    print("Q-Network updated done!")
    
    return None

