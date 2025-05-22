from transformers import BertTokenizer, RobertaTokenizer, BertConfig, RobertaConfig, BertModel, RobertaModel
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

from transformers import AutoTokenizer, AutoModelForCausalLM

import json
import argparse
import torch
import random
import numpy as np

def soft_update(target, source, tau):

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + (param.data * tau))

def backup_loading(model_name):

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  
        torch_dtype=torch.float16 
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

def get_transformers(args):

    tokenizer = {'bert': BertTokenizer, 'roberta': RobertaTokenizer}
    config = {'bert': BertConfig, 'roberta': RobertaConfig}
    models = {'bert': BertModel, 'roberta': RobertaModel}

    config_file = config[args.model_name].from_pretrained(args.model_name_or_path)

    return {
        'config': config_file,
        'model': models[args.model_name].from_pretrained(args.model_name_or_path, config=config_file),
        'tokenizer': tokenizer[args.model_name].from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    }

def get_args_sft():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--data_name', default='esc', type=str,
                        help="dataset name")
    parser.add_argument('--model_name', default='bert', type=str,
                        help="model name")
    parser.add_argument("--model_name_or_path", default='bert-large-uncased',
                        type=str, help="model name or path")
    parser.add_argument("--output_dir", default='sft_models', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--data_dir", default='data', type=str,
                        help="The data directory.")
    parser.add_argument("--cache_dir", default='/storage_fast/ydeng/plm', type=str,
                        help="The cache directory.")

    ## Other parameters
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval.")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--do_lower_case", action='store_false',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gpu', default="0 1", type=str,
                        help="Use CUDA on the device.")
    parser.add_argument("--per_gpu_train_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--num_train_epochs", default=10, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--warmup_steps", default=400, type=int,
                        help="Linear warmup over warmup_steps.")
    
    # Changed from "6e-6" (Default) to "1e-5"
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--local_rank", default=-1, type=int,
                        help="DDP requirement.")

    return parser.parse_args()

def get_args_train():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-seed', type=int, default=1, help='random seed.')
    parser.add_argument('--num_gpus', type=int, default=4, help='number of gpus.')
    parser.add_argument('--epochs', '-me', type=int, default=50000, help='the number of RL train epoch')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor.')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate.')

    parser.add_argument('--data_name', type=str, default='p4g', choices=['esc','cima','cb', 'p4g', 'extes'],
                        help='One of {esc, cima, cb}.')
    parser.add_argument('--system', type=str, default='vicuna', choices=['vicuna','chatgpt','llama2'],
                        help='One of {vicuna, chatgpt, llama2}.')
    parser.add_argument('--user', type=str, default='vicuna', choices=['vicuna','chatgpt','llama2'],
                        help='One of {vicuna, chatgpt, llama2}.')
    parser.add_argument('--critic', type=str, default='vicuna', choices=['vicuna','chatgpt','llama2'],
                        help='One of {vicuna, chatgpt, llama2}.')
    parser.add_argument('--sft_dir', default='sft', #../pretrain/outputs/best_pretrain.pt
                        type=str, help="Pretrain model path.")
    parser.add_argument('--max_turn', type=int, default=8, help='max conversation turn')
    parser.add_argument('--mode', type=str, default='train', help='the mode in [train, test]')
    parser.add_argument('--load_rl_epoch', type=int, default=0, help='load agent from epoch')


    parser.add_argument("--epsilon", type = float, default = 0.5, help="The balance of exploration or exploitation")
    parser.add_argument("--cache_dir", default='/storage_fast/ydeng/plm', type=str, help="The cache directory.")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model_path", type=str, default="Qwen2.5-14B")
    parser.add_argument("--model_name", type=str, default="roberta")
    parser.add_argument("--model_name_or_path", default='roberta-large', type=str, help="model name or path")

    parser.add_argument("--do_lower_case", action='store_false', help="Set this flag if you are using an uncased model.")

    parser.add_argument('--max_steps', type=int, default=1000, help='max training steps')
    parser.add_argument('--top_k', type=int, default=4, help='Top probabilities')
    parser.add_argument('--sample_times', type=int, default=2, help='the epoch of sampling')
    parser.add_argument('--eval_num', type=int, default=1, help='the number of steps to evaluate RL model and metric')
    parser.add_argument('--save_num', type=int, default=1, help='the number of steps to save RL model and metric')
    
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval.")

    return parser

def set_random_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def device_setup(args):
    
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
    
    devices_id = [int(device_id) for device_id in args.gpu.split()]

    device = (
        torch.device("cuda:{}".format(str(devices_id[0])))
        if use_cuda
        else torch.device("cpu")
    )

    return device, devices_id

def load_p4g():

    # Load the files of each separately
    dataset = {'train':[], 'test':[], 'dev':[]}
    for key in dataset:

        with open(f"data/P4G/api_annotated/{key}.json") as file:
            dataset[key] = json.load(file)

    return dataset

def load_extes():

    # Load the JSON file
    data = []
    with open('data/ExTES/ExTES.json') as file:
        data = json.load(file)

    # Divide into 10717/200/200
    random.shuffle(data)

    train_set = data[:10717]
    valid_set = data[10717:10917]
    test_set = data[10917:11117]

    return {
        'train': train_set, 
        'valid': valid_set, 
        'test': test_set
    }

def load_dataset(data_name):

    dataset = {'train':[], 'test':[], 'valid':[]}

    for key in dataset:
        with open("data/%s-%s.txt"%(data_name, key),'r') as infile:
            for line in infile:
                dataset[key].append(eval(line.strip('\n')))
                
    return dataset
