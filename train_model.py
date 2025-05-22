from fastchat.model import add_model_args
from tqdm import tqdm
from itertools import count

from llm_priors import train_qnetwork, soft_update
from q_adapter import QAdapter

from utils_data import get_action_list
from chip import CHIP
from misc import get_args_train, load_dataset, get_transformers, set_random_seed, load_p4g, load_extes
from env import Env

import numpy as np
import re
import torch
import json
import random
import pickle

def double_check_llm(next_action, full_options):

    return "Others" if next_action not in full_options else next_action

def self_play_evaluate(args, dataset, env, full_options, action_list, prob_network):
    test_env = None

    if env.is_llama_vicuna():
        test_env = Env(
            args, 
            dataset, 
            'test', 
            env.get_llm_model(), 
            env.get_llm_tokenizer()
        )

    success_rate, avg_turn, total_reward = 0., 0., 0.
    success_turn = [0] * args.max_turn

    test_size = test_env.get_dataset_size()

    all_conversation_data = []
    id_count = 1

    for test_num in tqdm(range(test_size)):

        epi_reward, done, is_last_turn = 0, 0, False
        state = test_env.reset()

        for t in count():

            # Perform the first step - LLM 
            state_action_pairs, action_choices = test_env.get_prior_actions_llm(state, action_list, full_options)

            # Encode with BERT -> Q-values -> Choose
            encoded_features = prob_network.transform_features(state_action_pairs)
            q_values = prob_network(encoded_features)

            # Choose the action
            current_action_id = q_values.argmax()
            action = action_choices[current_action_id]

            # If number not in, check if it is in the action list it self
            # Prevent hallucination
            action = double_check_llm(action, action_list)

            state, reward, done = test_env.perform_self_play(action, state)

            if (args.data_name == 'cb') and (reward < 0):
                reward = 0

            epi_reward += reward

            if done:

                total_reward += epi_reward
                avg_turn += t + 1

                if done == 1:
                    success_turn = [
                        v + 1 if i > t else v
                        for i, v in enumerate(success_turn)
                    ]
                    success_rate += 1

                # Store the conversation data
                all_conversation_data.append({
                    'id': id_count,
                    'conversation': state,
                })

                id_count += 1

                break
    
    # Save the file
    with open(f'saved_conversations/{args.data_name}_all.json', 'w') as f:
        json.dump(all_conversation_data, f, indent=4)
    
    success_mean = float(success_rate)/test_size
    average_turns = float(avg_turn)/test_size
    average_reward = total_reward/test_size

    full_metrics = [success_mean, average_turns, average_reward]

    return full_metrics

# Get the args
parser = get_args_train()
add_model_args(parser)
args = parser.parse_args()

# Load the dataset
dataset = {}

if args.data_name == 'p4g':
    dataset = load_p4g()
elif args.data_name == 'extes':
    dataset = load_extes()
else:
    dataset = load_dataset(args.data_name)

# Get the models
llm_dict = get_transformers(args)
set_random_seed(args.seed)

# Load the action list
action_list = list(get_action_list(args.data_name).keys())

full_options = [f"({i+1}) {action}\n" for i, action in enumerate(action_list)]
full_options = "".join(list(full_options))

# Load the Q adapter: One trainable, one fixed
dim_size = 768

prob_network = QAdapter(dim_size)
target_prob_network = QAdapter(dim_size)

prob_network.cuda()
target_prob_network.cuda()

soft_update(target_prob_network, prob_network, tau=1)

prob_optimizer = torch.optim.AdamW(prob_network.parameters(), lr=args.lr)

# Initialize environment
env = Env(args, dataset, 'train')

test_performance = []
buffer_list = []

# Total parameters
total_params = sum(p.numel() for p in prob_network.parameters())
print(f"Total params: {total_params}")

# Trainable parameters
trainable_params = sum(p.numel() for p in prob_network.parameters() if p.requires_grad)
print(f"Trainable params: {trainable_params}")

curr_decision_chosen = ''
next_decision_chosen = ''

# OPTIONAL: Set the sample size to dataset size
#args.sample_times = len(dataset['train'])

for train_step in range(1, args.max_steps + 1):

    success_rate, avg_turn, total_reward = 0., 0., 0.
    #loss = torch.tensor(0, dtype=torch.float, device=args.device)

    for i_episode in tqdm(range(args.sample_times), desc='sampling'):

        try:
            state = env.reset()
            epi_reward = 0
            done = False

            # Perform the first step - LLM Priors
            current_state_action_pairs, current_action_choices = env.get_prior_actions_llm(state, action_list, full_options)

            # Encode with BERT -> Q-values -> Choose
            current_encoded_features = prob_network.transform_features(current_state_action_pairs)
            q_values = prob_network(current_encoded_features)
            
            # Choose between exploitation or exploration
            current_action_id = 0
            exploration_val = random.uniform(0, 1)

            if exploration_val < args.epsilon:
                curr_decision_chosen = 'Greedy Epsilon'
                current_action_id = random.choice(range(len(current_action_choices)))
            else:
                curr_decision_chosen = 'Argmax'
                current_action_id = q_values.argmax()

            # Choose the action
            current_action = current_action_choices[current_action_id]

            for t in count():

                # Prevent hallucination
                current_action = double_check_llm(current_action, action_list)

                # Self-play time
                state, reward, done = env.perform_self_play(current_action, state)

                # Perform the next step - LLM Priors
                next_state_action_pairs, next_action_choices = env.get_prior_actions_llm(state, action_list, full_options)

                # Encode with BERT -> Q-values -> Choose
                next_encoded_features = prob_network.transform_features(next_state_action_pairs)
                next_q_values = prob_network(next_encoded_features)

                # Choose between exploitation or exploration
                next_action_id = 0
                exploration_val = random.uniform(0, 1)

                if exploration_val < args.epsilon:
                    next_decision_chosen = 'Greedy Epsilon'
                    next_action_id = random.choice(range(len(next_action_choices)))
                else:
                    next_decision_chosen = 'Argmax'
                    next_action_id = q_values.argmax()

                # Choose the action
                next_action = next_action_choices[next_action_id]

                # Store in the buffer
                buffer_list.append({
                    "feat": current_encoded_features[current_action_id, :].unsqueeze(0).detach().cpu().numpy(),
                    "feat_all": current_encoded_features.detach().cpu().numpy(),
                    "reward": np.array(reward, dtype=np.float32),
                    "done": np.array(done),
                    "next_feat": next_encoded_features.to(torch.float32).detach().cpu().numpy()
                })

                # Update the variables for chaining
                current_encoded_features = next_encoded_features
                current_action = next_action
                current_action_id = next_action_id
                current_action_choices = next_action_choices
                curr_decision_chosen = next_decision_chosen

                if done:

                    avg_turn += t + 1
                    total_reward += epi_reward

                    success_rate = (success_rate + 1) if done == 1 else success_rate

                    break
        except:
            continue
    
    # Train the Q-network
    train_qnetwork(buffer_list, prob_optimizer, prob_network, target_prob_network, args.device)
    

    all_sr = self_play_evaluate(args, dataset, env, full_options, action_list, prob_network)

    test_performance.append(all_sr)
    print(f"Scores at {train_step} are: {all_sr}")
