#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import time
import wandb
import argparse
import random
import sys
sys.path.insert(0, './GPT2ForwardBackward')

from nltk.corpus import stopwords
from opt_util import *
from util import *
from bleuloss import batch_log_bleulosscnn_ae

stop_words = set(stopwords.words('english'))


def options():
    parser = argparse.ArgumentParser()
    ## setting
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--no-cuda", action="store_true", help="no cuda")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--pretrained_model", type=str, default="Llama-2-7b-chat-hf")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--straight-through", action="store_true")
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--rl-topk", type=int, default=0)
    parser.add_argument("--lexical", type=str, default='max', choices=['max', 'ppl_max', 'all', 'bleu'])
    parser.add_argument("--lexical-variants", action="store_true", help="")
    parser.add_argument("--if-zx", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    ## experiment
    parser.add_argument("--input-file", type=str,
                        default="./data/lexical/commongen_data/test.multi.constraint.json")
    parser.add_argument("--version", type=str, default="")
    parser.add_argument("--start", type=int, default=1, help="loading data from ith examples.")
    parser.add_argument("--end", type=int, default=10, help="loading data util ith examples.")
    parser.add_argument("--repeat-batch", type=int, default=1, help="loading data util ith examples.")
    parser.add_argument("--mode", type=str, default='constrained_langevin',
                        choices=['suffix', 'control', 'paraphrase','proxy','proxy_one'])
    parser.add_argument("--control-type", type=str, default='sentiment', choices=['sentiment', 'lexical', 'style', 'format'])
    ## model
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--length", type=int, default=15, help="maximum length of optimized logits.")
    parser.add_argument("--max-length", type=int, default=50, help="maximum length of complete sentence.")
    parser.add_argument("--frozen-length", type=int, default=0, help="length of optimization window in sequence.")
    parser.add_argument("--goal-weight", type=float, default=0.1)
    parser.add_argument("--rej-weight", type=float, default=0.05)
    parser.add_argument("--abductive-filterx", action="store_true", help="filter out keywords included in x")
    parser.add_argument("--lr-nll-portion", type=float, default=1)
    parser.add_argument("--prefix-length", type=int, default=0, help="length of prefix.")
    parser.add_argument("--counterfactual-max-ngram", type=int, default=3)
    parser.add_argument("--no-loss-rerank", action="store_true", help="")
    parser.add_argument("--use-sysprompt", action="store_true", help="")
    # temperature
    parser.add_argument("--input-lgt-temp", type=float, default=1,
                        help="temperature of logits used for model input.")
    parser.add_argument("--output-lgt-temp", type=float, default=1,
                        help="temperature of logits used for model output.")
    parser.add_argument("--rl-output-lgt-temp", type=float, default=1,
                        help="temperature of logits used for model output.")
    parser.add_argument("--init-temp", type=float, default=0.1,
                        help="temperature of logits used in the initialization pass. High => uniform init.")
    parser.add_argument("--init-mode", type=str, default='original', choices=['random', 'original'])
    # lr
    parser.add_argument("--stepsize", type=float, default=0.1, help="learning rate in the backward pass.")
    parser.add_argument("--stepsize-ratio", type=float, default=1, help="")
    parser.add_argument("--stepsize-iters", type=int, default=1000, help="")
    # iterations
    parser.add_argument("--num-iters", type=int, default=1000)
    parser.add_argument("--min-iters", type=int, default=0, help="record best only after N iterations")
    parser.add_argument("--noise-iters", type=int, default=1, help="add noise at every N iterations")
    parser.add_argument("--win-anneal-iters", type=int, default=-1, help="froze the optimization window after N iters")
    parser.add_argument("--constraint-iters", type=int, default=1000,
                        help="add one more group of constraints from N iters")
    # gaussian noise
    parser.add_argument("--gs_mean", type=float, default=0.0)
    parser.add_argument("--gs_std", type=float, default=0.01)
    parser.add_argument("--large-noise-iters", type=str, default="-1", help="Example: '50,1000'")
    parser.add_argument("--large_gs_std", type=str, default="1", help="Example: '1,0.1'")
    # 代理模型地址
    parser.add_argument("--proxy_model", type=str, default="vicuna-7b-v1.5")
    parser.add_argument("--proxy_model_path", type=str)
    parser.add_argument("--kl_max_weight", type=float, default=100)
    parser.add_argument("--wandb_project", type=str)
    #强化学习
    parser.add_argument("--rl_eval_interval", type=int, default=200)
    parser.add_argument("--kl_threshold", type=float, default=100)
    parser.add_argument("--forbidden_threshold", type=float, default=0.8)
    parser.add_argument("--cw_weight", type=float, default=100)
    parser.add_argument("--cw_loss_kappa", type=float, default=0.05)

    args = parser.parse_args()
    return args

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def main():
    args = options()
    # 设置 GPU 0
    if torch.cuda.is_available():
        # 获取可用的 GPU 数量
        available_gpus = torch.cuda.device_count()

        # 如果有可用的 GPU，选择第一个可用的 GPU
        if available_gpus > 0:
            # 自动选择第一个可用的 GPU
            torch.cuda.set_device(0)
            print(f"Using GPU {torch.cuda.current_device()}: {torch.cuda.get_device_name(0)}")
        else:
            print("No GPUs are available.")
    else:
        print("CUDA is not available, using CPU instead.")


    device = "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    # 根据操作系统设置基础路径
    if os.name == 'nt':  # Windows 系统
        base_dir = r"D:\ZLCODE\model"
    else:  # Linux 或其他系统
        base_dir = "/home/zl/ZLCODE/model"  # 请将此处修改为 Linux 下的模型存放路径

    # 利用 os.path.join 拼接完整路径
    model_path_dicts = {
        "Llama-2-7b-chat-hf": os.path.join(base_dir, "Llama-2-7b-chat-hf"),
        "Vicuna-7b-v1.5": os.path.join(base_dir, "vicuna-7b-v1.5"),
        "guanaco-7b": os.path.join(base_dir, "guanaco-7B-HF"),
        "mistral-7b": os.path.join(base_dir, "Mistral-7B-Instruct-v0.2"),
    }


    model_path = model_path_dicts[args.pretrained_model]
    print("model_path:", model_path, type(model_path))
    if args.seed != -1:
        seed_everything(args.seed)
    # Load pretrained model
    if "proxy" not in args.mode:
        model, tokenizer = load_model_and_tokenizer(model_path,
                                                low_cpu_mem_usage=True,
                                                use_cache=False,
                                                device=device)

        # Freeze GPT-2 weights
        model.eval()
        for param in model.parameters():
            param.requires_grad = False





    print(args.mode)
    if "suffix" in args.mode:
        from attack_suffix import attack_generation
        attack_generation(model, tokenizer, device, args)
    elif "paraphrase" in args.mode:
        from attack_paraphrase import attack_generation
        attack_generation(model, tokenizer, device, args)
    elif "control" in args.mode:
        from attack_control import attack_generation
        attack_generation(model, tokenizer, device, args)
    elif "proxy_one" in args.mode:
        from attack_suffix_proxy_one import attack_generation   ##再添加一个代理模型
        attack_generation(model_path, device, args)
    elif "proxy" in args.mode:
        from attack_suffix_proxy import attack_generation   ##再添加一个代理模型
        attack_generation(model_path, device, args)


if __name__ == "__main__":
    main()
