import torch
import torch.nn.functional as F
import numpy as np
import time
import wandb
import logging
import os
from datetime import datetime

from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from evaluation.bert_score import score
from transformers import AutoModelForCausalLM, AutoTokenizer
from util import *

import time
import seaborn as sns
import copy

from matplotlib import pyplot as plt

from tqdm import tqdm
#新添加的import
from model.huggingface import FineTuneConfig

stop_words = set(stopwords.words('english'))


class SuffixManager:
    """后缀管理器类"""
    def __init__(self, tokenizer, use_system_instructions=False, conv_template=None):
        self.tokenizer = tokenizer
        self.use_system_instructions = use_system_instructions
        self.conv_template = conv_template
        
    def get_prompt(self, prompt_str: str, system_str: str = "") -> str:
        """获取完整提示词"""
        if self.use_system_instructions:
            return f"{system_str}\n{prompt_str}"
        return prompt_str

# 设置日志记录器
def setup_logger(args):
    # 创建logs目录（如果不存在）
    log_dir = os.path.join('outputs', 'Llama-2-7b-chat-hf', 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 创建一个带时间戳的日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'decode_{timestamp}.log')
    
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        encoding='utf-8',
        handlers=[
            logging.FileHandler(log_file,encoding = 'utf-8')
           # logging.StreamHandler()  #同时输出到控制台

        ]
    )
    return logging.getLogger()

##加载代理模型
def load_proxy_model(
    model_path: str,
    use_system_instructions: bool = False,
    ft_config: FineTuneConfig | None = None,
    device: str = 'cuda',
    **kwargs
) -> tuple:
    """
    加载用于后缀攻击的代理模型
    
    Args:
        model_path (str): 模型路径或名称
        use_system_instructions (bool): 是否使用系统指令
        ft_config (FineTuneConfig): 微调配置
        device (str): 运行设备
        **kwargs: 其他参数
    
    Returns:
        tuple: (model, tokenizer, suffix_manager)
    """
    try:
        # 检查GPU可用性
        if device == 'cuda' and not torch.cuda.is_available():
            print("警告: 未检测到GPU,切换到CPU模式")
            device = 'cpu'
            
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side='left'
        )
        
        # 设置填充标记
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device=='cuda' else torch.float32,
            device_map='auto' if device=='cuda' else None
        )
        
        # 如果不需要微调,冻结模型参数
        if ft_config is None:
            for param in model.parameters():
                param.requires_grad = False
        
        # 将模型移动到指定设备
        if device == 'cpu':
            model = model.to(device)
            
        # 设置为评估模式
        model.eval()
        
        # 创建后缀管理器
        suffix_manager = SuffixManager(
            tokenizer=tokenizer,
            use_system_instructions=use_system_instructions
        )
        
        return model, tokenizer, suffix_manager
        
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        raise

def decode(model, tokenizer, device, x="", z="", constraints=None, args=None, sys_prompt=None, prefix=None, model_back=None, zz=None):
    '''
    x: left context   (prompt in lexical lexical task)
    z: optimization target  (original ending in counterfactual task)
    constraints: (constraint set in lexical constrained task)
    '''
    model.eval()   #设置评估模式
    # 设置日志记录器
    logger = setup_logger(args)
    #不良词汇
    words = ['sorry', 'apologetic', 'regretful',
              'unethical', 'immoral', 'dishonest',
              'illegal', 'unlawful', 'illicit',
              'harmful', 'detrimental', 'injurious',
              'afraid', 'fearful', 'apprehensive',
              'cannot', 'unable', 'unable to',
              'apologize', 'regret',
              'unable', 'incapable', 'powerless',
              'not', 'no', 'nay',
              'responsible', 'accountable', 'liable',
              'ethic', 'morality', 'moral',
              'legal', 'lawful', 'legitimate',
              'ethical', 'moral', 'principled',
              'fulfill', 'accomplish', 'achieve',
              'just', 'fair', 'equitable',
              'trustworthy', 'reliable', 'dependable',
              'repentant', 'penitent', 'remorseful',
              'unprincipled', 'unscrupulous', 'corrupt',
              'unlawful', 'prohibited', 'forbidden',
              'noxious', 'destructive', 'damaging',
              'timid', 'terrified', 'panicked',
              'unable', 'incompetent', 'incapacitated',
              'prohibited', 'banned', 'forbidden',
              'deny', 'refuse', 'decline',
              'answerable', 'culpable', 'answerable',
              'moral code', 'values', 'principles',
              'legitimate', 'valid', 'authorized',
              'virtuous', 'righteous', 'honorable',
              'satisfy', 'complete', 'execute',
              'fair-minded', 'impartial', 'equitable',
              'reliable', 'trustable', 'faithful', 'invalid','safe', 'not', "can't", "but", "against"]

    #包含大写小写的不良词汇
    lowercase_words = [word.upper() for word in words]

    bad_words = words + lowercase_words
    
    bad_words = ' '.join(bad_words)

    BIG_CONST = 1e10
    if args.use_sysprompt:
        x_sys = sys_prompt + x
        x_ = tokenizer.encode(x_sys)[1:]
    else:
        x_ = tokenizer.encode(x)[1:]
    x_t = torch.tensor(x_, device=device, dtype=torch.long)
    x_onehot = one_hot(x_t, dimension=tokenizer.vocab_size)

    # repeat batch_size times
    x_t = x_t.unsqueeze(0).repeat(args.batch_size, 1)
    x_onehot = x_onehot.repeat(args.batch_size, 1, 1)

    z_mask = None
    x_mask = None
    # extract keywords:
    z_ = tokenizer.encode(z)[1:]  
    z_t = torch.tensor(z_, device=device, dtype=torch.long)

    z_onehot = one_hot(z_t, dimension=tokenizer.vocab_size)
    z_onehot = z_onehot.repeat(args.batch_size, 1, 1)

    z_t = z_t.unsqueeze(0).repeat(args.batch_size, 1)

    length = args.length
    if length <= 0:
        length = z_t.shape[1] - length
    if args.verbose:
        logger.info("x:\t|%s|\nz:\t|%s|\nlength:\t%d\nconstraints:\t%s" % (
            tokenizer.decode(x_), tokenizer.decode(z_), length, constraints))
    
    # z_mask: [batch_size, vocab_size]
    z_words = word_tokenize(z[:])  
    z_nonstop_words = [w.lower() for w in z_words if w.lower() not in stop_words and w.isalnum()]
    z_nonstop_words += [z_words[0]]  # add the first token
    z_nonstop_words = ' ' + ' '.join(z_nonstop_words)
    z_nonstop_ = tokenizer.encode(z_nonstop_words)
    logger.info('|' + z_nonstop_words + '|')

    z_mask = np.zeros([tokenizer.vocab_size])
    z_mask[z_nonstop_] = 1.
    z_mask = torch.tensor(z_mask, device=device)
    z_mask = z_mask.unsqueeze(0).unsqueeze(0).repeat(args.batch_size, length, 1)

    ###################################################

    length = args.length
    if length <= 0:
        length = x_t.shape[1] - length

    x_words = tokenizer.encode(bad_words)
    x_mask = np.zeros([tokenizer.vocab_size])
    x_mask[x_words] = 1.
    x_mask = torch.tensor(x_mask, device=device)

    bad_mask = x_mask.unsqueeze(0).unsqueeze(0).repeat(args.batch_size, length, 1)

    bad_mask = torch.ones_like(bad_mask, device=device) - bad_mask

    bad_words_ = tokenizer.encode(bad_words)[:]  # delete the "." token we appended before
    bad_words_t = torch.tensor(bad_words_, device=device, dtype=torch.long)

    bad_words_onehot = one_hot(bad_words_t, dimension=tokenizer.vocab_size)
    bad_words_onehot = bad_words_onehot.repeat(args.batch_size, 1, 1)

    bad_words_t = bad_words_t.unsqueeze(0).repeat(args.batch_size, 1)


    ###################################################

    # 根据初始化模式选择初始logits
    if args.init_mode == 'original':
        # 使用模型初始化logits unitl中  x_t已经变成批次的输入了
        init_logits = initialize(model, x_t, length, args.init_temp, args.batch_size ,device, tokenizer)
    else:
        # 使用one-hot向量初始化logits
        init_logits = z_onehot / 0.01  # 缩放one-hot向量
        init_logits = init_logits[:, :length, :]
        # 如果需要的长度大于初始logits，则用零填充
        if length > init_logits.shape[1]:
            init_logits = torch.cat(
                [init_logits,
                 torch.zeros([args.batch_size, length - init_logits.shape[1], tokenizer.vocab_size], device=device)],
                dim=1)
    
    # 从初始logits生成文本并打印 util.get_text_from_logits根据使用模型的tokenizer
    text, _, _ = get_text_from_logits(init_logits, tokenizer)
    for bi in range(args.batch_size):
        logger.info("[initial]: %s" % (text[bi]))
    
    # 如果启用了wandb，初始化wandb项目
    if args.wandb:
        run_name = f"{args.mode}_{int(round(time.time() * 1000))}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=args,
            reinit=True)  # 确保每次都重新初始化

    # 将初始logits赋值给y_logits作为优化目标
    y_logits = init_logits

    # 创建优化的扰动参数epsilon
    epsilon = torch.nn.Parameter(torch.zeros_like(y_logits, dtype=torch.float32))  #创建一个于y_logits维度相同的参数全零张量
    if args.prefix_length > 0:
        pass
    else:
        # 创建Adam优化器
        optim = torch.optim.Adam([epsilon], lr=args.stepsize) #学习率优化器
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=args.stepsize_iters,
                                                gamma=args.stepsize_ratio) #学习率调度器

    frozen_len = args.frozen_length  #冻结的长度

    y_logits_ = None

    noise_std = 0.0

    # 确保prefix_length <= 0
    assert args.prefix_length <= 0, "The current code does not support prefix-length > 0"

    # 准备模型输入
    soft_forward_x = x_onehot[:, -1:, :]
    if x_t.shape[1] == 1:
        x_model_past = None
    else:
        # 获取模型的past key values用于加速生成
        x_model_outputs = model(x_t[:, :-1], use_cache=True)
        x_model_past = x_model_outputs.past_key_values

    mask_t = None

    # 主优化循环
    for iter in tqdm(range(args.num_iters), desc="Processing Goals"):
        # 你的逻辑代码
        print(f"Processing iteration {iter}")
        optim.zero_grad()  # 清除梯度

        # 将扰动加到logits上
        y_logits_ = y_logits + epsilon
        
        
        soft_forward_y = y_logits_ / 0.001  # 温度缩放
        
        
        # 直通估计器(Straight-through estimator)处理 为了前向传播
        if args.straight_through:
            if mask_t is None:
                soft_forward_y = (y_logits_.detach() / 0.001 - y_logits_).detach() + y_logits_
            else:
                soft_forward_y = top_k_filter_3d(y_logits_, args.topk, mask=mask_t, extra_mask=x_mask, bad_mask=None) / 0.001
            
        
        # 使用模型前向传播
        if args.fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                y_logits_t = soft_forward(model, soft_forward_x, soft_forward_y, args.topk, extra_mask=x_mask, x_past=x_model_past, bad_mask=None)
        else:
            y_logits_t = soft_forward(model, soft_forward_x, soft_forward_y, args.topk, extra_mask=x_mask, x_past=x_model_past, bad_mask=None)
        

        # 生成top-k mask
        if args.topk == 0:
            mask_t = None
        else:
            _, indices_t = torch.topk(y_logits_t, args.topk)
            mask_t = torch.zeros_like(y_logits_t).scatter_(2, indices_t, 1)
        
        # 计算流畅性损失
        flu_loss = soft_nll(
            top_k_filter_3d(y_logits_t / args.output_lgt_temp, args.topk, extra_mask=x_mask, bad_mask=None),
            y_logits_ / args.input_lgt_temp)

        # 计算xyz序列的logits
        soft_forward_y_ = (y_logits_.detach() / 0.001 - y_logits_).detach() + y_logits_
        if args.fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                xyz_logits, xy_length = soft_forward_xyz(model, soft_forward_x, soft_forward_y_, z_onehot)
        else:
            xyz_logits, xy_length = soft_forward_xyz(model, soft_forward_x, soft_forward_y_, z_onehot)

        # 重塑张量维度
        bz = args.batch_size
        lg = xyz_logits.shape[1]
        st = xy_length - 1
        ed = xyz_logits.shape[1] - 1
        xyz_logits = xyz_logits.view(-1, xyz_logits.shape[-1])
        z_logits = torch.cat([xyz_logits[bi * lg + st:bi * lg + ed, :] for bi in range(bz)], dim=0)


        #能量函数
        # 计算目标损失
        c_loss_1 = torch.nn.CrossEntropyLoss(reduction='none')(
            z_logits,
            z_t.view(-1))
        c_loss_1 = c_loss_1.view(args.batch_size, -1).mean(-1)

        # 计算BLEU损失（用于避免生成不良词）
        c_loss_2 = batch_log_bleulosscnn_ae(
            decoder_outputs=y_logits_.transpose(0, 1),
            target_idx=bad_words_t,
            ngram_list=[1]
        )


        # 组合总损失
        loss = args.goal_weight * c_loss_1 + 1 * flu_loss - args.rej_weight * c_loss_2
        loss = loss.mean()
        # logger.info(f"  - Total Loss: {loss.item()}")
        
        if args.wandb:
            # 记录损失和范数到wandb
            wandb_step = iter + 1  # 使用1-based的step计数
            wandb.log({
                'loss/total': loss.item(),
                'loss/fluency': flu_loss.mean().item(),
                'loss/target': c_loss_1.mean().item(),
                'loss/bleu': c_loss_2.mean().item(),
                'norm/epsilon': torch.norm(epsilon).item(),
                'norm/y_logits': torch.norm(y_logits).item(),
                'norm/soft_forward_y': torch.norm(soft_forward_y).item(),
                'norm/y_logits_t': torch.norm(y_logits_t).item(),
                'learning_rate': scheduler.get_last_lr()[0]
            }, step=wandb_step)
        
        # 如果不是最后一次迭代，进行反向传播和优化
        if iter < args.num_iters - 1: 
            loss.backward()
            
            if args.wandb:
                # 记录梯度到wandb，使用相同的step
                wandb.log({
                    'grad/epsilon': torch.norm(epsilon.grad).item()
                }, step=wandb_step)  # 使用相同的step
            
            # 记录梯度信息
            # logger.info(f"\n[Iter {iter+1}] Gradient norms:")
            # logger.info(f"  - Epsilon grad norm: {torch.norm(epsilon.grad).item()}")   #计算梯度的L2范数
            
            optim.step()  #优化器更新
            scheduler.step()  #学习率调度
            last_lr = scheduler.get_last_lr()[0]
            # logger.info(f"[Iter {iter+1}] Learning rate: {last_lr}")
            
        # 定期打印生成结果
        if args.verbose and ((iter + 1) % args.print_every == 0 or iter == 0 or iter + 1 == args.num_iters):
            text, _, last_text_ids = decode_with_model_topk(
                model, y_logits_, args.topk, soft_forward_x, x_model_past, tokenizer, extra_mask=None, bad_mask=None)
            text_post = text
            for bi in range(args.batch_size):
                prompt = x + " " + text_post[bi]
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                logger.info("\n Output of the model:\n")
                output_ids  = model.generate(inputs=input_ids, temperature=0.7, max_length = 512, do_sample=True, top_k=args.topk)
                
                logger.info(tokenizer.decode(output_ids[0], skip_special_tokens=True))
                logger.info(
                    "%d, loss: %.4f,flu_loss: %.4f, c_loss_1: %.4f, c_loss_2: %.4f, lr: %.4f, |%s|" % (
                        iter + 1, loss.item(), flu_loss[bi].item(), c_loss_1[bi].item(), c_loss_2[bi].item(), last_lr, text_post[bi]))
                        
        # 添加噪声以增加多样性
        if iter < args.num_iters - 1:
            # 解析噪声参数
            large_noise_iters = [int(_) for _ in args.large_noise_iters.split(',')]
            large_gs_stds = [float(_) for _ in args.large_gs_std.split(',')]
            noise_std = 0.
            
            # 根据迭代次数决定噪声大小
            if iter % args.noise_iters == 0:
                noise_last = True
                for ni in range(len(large_noise_iters)):
                    if iter < large_noise_iters[ni]:
                        noise_last = False
                        break
                if noise_last:
                    noise_std = args.gs_std
                else:
                    noise_std = large_gs_stds[ni]

                # 生成并应用噪声
                noise = torch.normal(mean=args.gs_mean, std=noise_std, size=epsilon.size(),
                                     device='cuda', requires_grad=False)
                if args.win_anneal_iters >= 0 and iter >= args.win_anneal_iters:
                    zeros = torch.zeros_like(noise)
                    noise_mix = torch.cat([zeros[:, :frozen_len], noise[:, frozen_len:]], dim=1)
                    y_logits = y_logits + noise_mix
                else:
                    y_logits = y_logits + noise

    # 最终生成文本(提示词文本)
    text, _, last_text_ids = decode_with_model_topk(
        model, y_logits_, args.topk, soft_forward_x, x_model_past, tokenizer, extra_mask=None, bad_mask=None)

    text_post = text
    decoded_text = []
    # 对每个batch生成完整文本
    for bi in range(args.batch_size):
        prompt = x + " " + text_post[bi]
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output_ids  = model.generate(inputs=input_ids, temperature=0.7, max_length = 512, pad_token_id=tokenizer.pad_token_id, do_sample=True, top_k=args.topk)
        output_ids = output_ids[:, input_ids.shape[1]:]
        text_dec = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        decoded_text.append(text_dec.strip())
    
    # 计算最终的困惑度得分
    last_rank_loss = model(input_ids=last_text_ids, labels=last_text_ids).loss
    last_rank_loss = last_rank_loss.detach().clone().data.cpu().numpy()
    ppl_last = np.exp(last_rank_loss)  #这个是代表一个批次的平均困惑度
    ppl=[]
    for bi in range(args.batch_size):
        ppl.append(ppl_last)
    # 组合最终的提示和生成文本
    prompt_with_adv = [x + " " + t for t in text_post]
    
    return ppl, text, text_post, decoded_text, prompt_with_adv