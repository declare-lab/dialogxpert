from fastchat.model import load_model, get_conversation_template
from prompt import *
from qwen_prompts import *
from misc import set_random_seed, backup_loading

import openai
import torch
import nltk
import re
import numpy as np

class Env(object):

    def __init__(self, args, dataset, mode, env_model = None, env_tokenizer = None):

        # Vicuna and LLama check
        llm_players = [args.system, args.user, args.critic]

        self._is_vicuna = 'vicuna' in llm_players
        self._is_llama = 'llama2'in llm_players
        self._gpt_api_key = ''

        if (self._is_vicuna or self._is_llama):

            self._llm_model = env_model
            self._llm_tokenizer = env_tokenizer

            # For training, load from scratch
            if mode == 'train':

                self._llm_model, self._llm_tokenizer = backup_loading(args.model_path)
                

        self._args = args
        self._dataset = dataset[mode]
        self._mode = mode

        self._max_turn = args.max_turn

        self._conversation = []
        self._user_emotions = []
        self._cur_conv_step = 0
        self._top_k = args.top_k

        # For test dataset
        self._test_num = 0

        self._reward_dict = {
            'esc': {
                'worse': -1.0,
                'same': -0.5,
                'better': 0.5,
                'solved': 1.0
            },

            'cima': {
                'incorrect': -1.0,
                'did not': -0.5,
                'part': 0.5,
                'whole': 1.0,
            },

            'p4g': {
                'refused': -1.0,
                'neutral': -0.5,
                'positive': 0.1,
                'agree': 1.0
            },

            'extes': {
                'worse': -1.0,
                'same': -0.5,
                'solved': 1.0
            }
        }

        self._message_format = {
            'esc': ESConvMessages, 
            'cima': CIMAMessages, 
            'cb': CBMessages
        }

        self._system_role = {
            'esc':'Therapist', 
            'cima': 'Teacher', 
            'cb': 'Buyer',
            'extes': 'Therapist',
            'p4g': 'Persuader'
        }

        self._user_role = {
            'esc':'Patient', 
            'cima': 'Student', 
            'cb': 'Seller',
            'extes': 'Patient',
            'p4g': 'Persuadee'
        }

        self._case = []

        set_random_seed(args.seed)

    def get_llm_model(self):
        return self._llm_model
    
    def get_llm_tokenizer(self):
        return self._llm_tokenizer

    def reset(self):

        self._cur_conv_step = 0
        self._conversation = []
        self._user_emotions = []

        if self._mode == 'train':
            self._case = np.random.choice(self._dataset)
        else:
            self._case = self._dataset[self._test_num]
            self._test_num += 1

        dataset = self._args.data_name

        # P4G
        if dataset == 'p4g':
            self._conversation.extend([
                {
                    "role": "Persuader",
                    "content": self._case['dialog'][0]['text']
                },
                {
                    "role": "Persuadee",
                    "content": self._case['dialog'][1]['text']
                }
            ])

            return self._conversation

        # ExTES
        if dataset == 'extes':
            return [
                {
                    "role": "Patient",
                    "content": self._case['description']
                }
            ]

        # ESConv
        if dataset == 'esc':

            self._user_emotions.append(self._case['emotion_type'])

            return [
                {
                    "role":"Patient", 
                    "content": self._case['situation']
                }
            ]

        # CIMA
        if dataset == 'cima':
            return [
                {
                    "role":"Teacher", 
                    "content":self._case['dialog'][0]['text']
                }, 
                {
                    "role":"Student", 
                    "content":self._case['dialog'][1]['text']
                }
            ]
        
        # CB
        return [
            {
                "role":"Buyer", 
                "content":"Hi, how much is the %s?" % self._case['item_name']
            }, 
            {
                "role":"Seller", 
                "content":"Hi, this is a good %s and its price is %s." % (self._case['item_name'], self._case['seller_price'])
            }
        ]
    
    def is_llama_vicuna(self):
        return True#self._is_llama or self._is_vicuna
    
    def get_dataset_size(self):
        return len(self._dataset)
    
    def get_prior_actions_llm(self, state, action_list, options):

        # Load the state -> full conversation
        full_conv = get_full_conversation(state)

        # Choose the action list from LLM
        prompt = ExTES_prompt(self._user_emotions, full_conv, options)

        # Prepare the input: Text to Token IDs
        formatted_prompt = self._llm_tokenizer.apply_chat_template(
            prompt, 
            tokenize=False, 
            add_generation_prompt=True
        )

        encoded_input = self._llm_tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(self._llm_model.device)

        # Get the output
        output_ids = []

        with torch.no_grad():

            output_ids = self._llm_model.generate(
                **encoded_input,
                max_new_tokens=25,

                # Sampling
                temperature = 1.0,
                top_p = 1.0,
                do_sample = True,
                early_stopping=True,
                
                # Output
                return_dict_in_generate = True,
                output_scores = True,
                output_hidden_states = True,

                # Caching
                use_cache=True
            )
        
        # Get the output
        pre_decoded_sequences = self._llm_tokenizer.batch_decode(
            output_ids.sequences
        )[0]

        output = pre_decoded_sequences.split('assistant')[-1]
        num_options = re.findall(r'\d+', output)
        num_options = list(map(int, num_options))

        # Translate numbers to action
        action_pairs = [action_list[num - 1] for num in num_options]

        # Create the state-action pairs
        state_action_pairs = [
            f"Action: {action} for Observation: {full_conv}"
            for action in action_pairs
        ]

        return state_action_pairs, action_pairs
    
    def predict_next_action(self, state, options):

        # Build input
        full_conv = get_full_conversation(state)

        # Load the prompt
        prompt = ExTES_prompt(self._user_emotions, full_conv, options)

        formatted_prompt = self._llm_tokenizer.apply_chat_template(
            prompt, 
            tokenize=False, 
            add_generation_prompt=True
        )

        input_ids = self._llm_tokenizer(formatted_prompt, return_tensors="pt").input_ids

        # Get the output
        output_ids = []

        with torch.no_grad():

            output_ids = self._llm_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=100,
                temperature = 1.0,
                do_sample = True,
                top_p=1.0,
                eos_token_id=self._llm_tokenizer.eos_token_id,
            )

        output_ids = output_ids[0][len(input_ids[0]):]

        output = self._llm_tokenizer.decode(
            output_ids, skip_special_tokens=True,
            spaces_between_special_tokens=False
        )

        return output

    def get_response(self, prompt, conv_role="assistant"):

        temperature = 1.0 if self._mode == 'test' else 0.7

        formatted_prompt = self._llm_tokenizer.apply_chat_template(
            prompt, 
            tokenize=False, 
            add_generation_prompt=True
        )

        input_ids = self._llm_tokenizer(formatted_prompt, return_tensors="pt").input_ids

        # Get the output
        output_ids = []

        with torch.no_grad():

            output_ids = self._llm_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens = self._args.max_new_tokens,
                temperature = temperature,
                early_stopping = True,
                do_sample = False
            )

        output_ids = output_ids[0][len(input_ids[0]):]

        output = self._llm_tokenizer.decode(
            output_ids, skip_special_tokens=True,
            spaces_between_special_tokens=False
        )

        return output
    
    def prepare_critic_output(self, conv_role="assistant"):

        prompt = ExTES_roleplay(self._case, 'critic', self._conversation, self._user_emotions)
        
        formatted_prompt = self._llm_tokenizer.apply_chat_template(
            prompt, 
            tokenize=False, 
            add_generation_prompt=True
        )

        input_ids = self._llm_tokenizer(formatted_prompt, return_tensors="pt").input_ids

        # Get the verdict
        output_ids = []

        with torch.no_grad():
            
            output_ids = self._llm_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=25,
                temperature = 1.1,
                do_sample = True,
                early_stopping=True,
                num_return_sequences=10,
            )

        # Return!
        outputs = []

        for o in output_ids:
            output_id = o[len(input_ids[0]):]


            output = self._llm_tokenizer.decode(
                output_id, skip_special_tokens=True,
                spaces_between_special_tokens=False
            )

            outputs.append(output)

        return outputs
    
    def calculate_reward(self, outputs, case):

        # Parse based on dataset
        if self._args.data_name in ['esc','cima', 'p4g', 'extes']:

            rewards = []

            for output in outputs:

                for key in self._reward_dict[self._args.data_name]:

                    if key in output.lower():
                        rewards.append(self._reward_dict[self._args.data_name][key])
                        break
            

            reward = (sum(rewards)/len(rewards)) if len(rewards) != 0 else 0
            
            return reward

        deals = []
        rewards = []

        for output in outputs:

            if 'have not' in output.lower():
                deals.append(-1)

            if 'have reached' in output.lower():
                deals.append(1)
            
            prices = re.findall(r"[-+]?\d*\.?\d+", output.replace(",",""))
            
            if len(prices) > 0:
                deal_price = float(prices[0])
                reward = (deal_price - case['seller_price']) / (case['buyer_price'] - case['seller_price'])
                rewards.append(reward)

        if -1 in deals:
            return -0.1

        reward = max(set(rewards), key = rewards.count) if len(rewards) != 0 else 0

        return reward

    def perform_self_play(self, action, conversation):

        # Flow: Prepare input -> Get response -> Process response

        # Self-play role: System -> Patient -> Critic

        # System
        messages = ExTES_roleplay(self._case, 'system', conversation, None, action)
        response = self.get_response(messages, conv_role="assistant")

        # Find the emotion and response
        emotion_start = response.find('Emotion:')
        emotion_end = response.find('Response:')
        
        # 8 -> Size of the string 'Emotion: '
        # 9 -> Size of the string 'Response: '
        emotion = response[emotion_start + 8 : emotion_end].strip()
        sys_reply = response[emotion_end + 9 :].strip()
        self._user_emotions.append(emotion.lower())

        response = self.postprocess_response(response,  self._user_role[self._args.data_name])
        self._conversation.append({"role": self._system_role[self._args.data_name], "content":sys_reply})

        # User
        messages = ExTES_roleplay(self._case, 'user', conversation, self._user_emotions)
        response = self.get_response(messages, conv_role="assistant")

        response = self.postprocess_response(response,  self._user_role[self._args.data_name])
        self._conversation.append({"role": self._user_role[self._args.data_name], "content":response})

        # Critic check and reward calculation
        messages = self.prepare_critic_output()
        reward = self.calculate_reward(messages, self._case)

        dataset = self._args.data_name

        # Reward breakdown
        done = 0

        if (dataset == 'extes') and (reward >= 0.5):
            done = 1

        if (dataset == 'p4g') and (reward > 0.6):
            done = 1

        if (dataset == 'esc') and (reward >= 0.5):
            done = 1

        if (dataset == 'cima') and (reward == 1):
            done = 1

        if (dataset == 'cb') and (reward >= 0):
            done = 1

        if self._cur_conv_step == (self._max_turn - 1):
            done = -1

        self._cur_conv_step += 1

        return self._conversation, reward, done
        
    def step(self, action):

        done = 0

        # Flow: Prepare input -> Get response -> Process response

        # User input
        messages = self._message_format[self._args.data_name](self._case, 'user', self._conversation)
        response = self.generate_response(self._args.system, messages, self._system_role[self._args.data_name])
        response = self.postprocess_response(response,  self._user_role[self._args.data_name])
        self._conversation.append({"role": self._user_role[self._args.data_name], "content":response})

        # System input
        messages = self._message_format[self._args.data_name](self._case, 'system', self._conversation, action)
        response = self.generate_response(self._args.system, messages, self._system_role[self._args.data_name])
        response = self.postprocess_response(response,  self._user_role[self._args.data_name])
        self._conversation.append({"role":  self._system_role[self._args.data_name],"content":response})

        # Critic input
        messages = self._message_format[self._args.data_name](self.case, 'critic', self.conversation)
        reward = self.calculate_reward(self._args.critic, messages, self.case)

        dataset = self._args.data_name

        if (dataset == 'esc') and (reward > 0.5):
            done = 1

        if (dataset == 'cima') and (reward == 1):
            done = 1

        if (dataset == 'cb') and (reward >= 0):
            done = 1

        if self._cur_conv_step == (self._max_turn - 1):
            done = -1

        self._cur_conv_step + 1

        return self._conversation, reward, done
    
    def postprocess_response(self, response, role):

        if role in response:
            response = response.split(role)[0].strip()

        sents = nltk.sent_tokenize(response)

        if len(sents) == 1:

            if response[-1] not in ['.','!','?',':']:
                return response + '.'
            
            return response.strip()
        try:

            if sents[-1].strip()[-1] not in ['.','!','?',':']:
                return ' '.join(sents[:-1]).strip()
            else:
                return response.strip()
        except Exception as e:
            return response.strip()
        
    def generate_response(self, model, messages, role):

        temperature = 0 if self._mode == 'test' else 0.7
        
        # Vicuna
        if model == 'vicuna':
            prompt = vicuna_prompt(messages, role)

            input_ids = self._llm_tokenizer([prompt]).input_ids
            max_new_tokens = self._args.max_new_tokens

            output_ids = self._llm_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=max_new_tokens,
                temperature = temperature,
                early_stopping=True
            )

            output_ids = output_ids[0][len(input_ids[0]):]

            output = self._llm_tokenizer.decode(
                output_ids, skip_special_tokens=True,
                spaces_between_special_tokens=False
            )

            return output

        return output
    
    def compute_reward(self, model, messages, case):

        # Get the reward based on the LLM
        if model == 'vicuna':

            prompt = vicuna_prompt(messages, 'critic')
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=16,
                temperature = 1.1,
                do_sample = True,
                early_stopping=True,
                num_return_sequences=10,
            )

            outputs = []

            for o in output_ids:
                output_id = o[len(input_ids[0]):]
                output = self.vicuna_tokenizer.decode(output_id, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
                outputs.append(output)

        # Parse based on dataset
        if self._args.data_name in ['esc','cima', 'p4g', 'extes']:

            rewards = []
            
            # Search based on keywords
            for output in outputs:
                
                # Search based on the reward dictionary
                for key in self.reward_dict[self._args.data_name]:

                    if key in output.lower():
                        rewards.append(self.reward_dict[self._args.data_name][key])
                        break
            
            reward = (sum(rewards)/len(rewards)) if len(rewards) != 0 else 0
            return reward

       
        deals = []
        rewards = []

        for output in outputs:

            if 'have not' in output.lower():
                deals.append(-1)

            if 'have reached' in output.lower():
                deals.append(1)
            
            prices = re.findall(r"[-+]?\d*\.?\d+", output.replace(",",""))
            
            if len(prices) > 0:
                deal_price = float(prices[0])
                reward = (deal_price - case['seller_price']) / (case['buyer_price'] - case['seller_price'])
                rewards.append(reward)

        if -1 in deals:
            return reward -0.1

        reward = max(set(rewards), key = rewards.count) if len(rewards) != 0 else 0

        return reward

    
