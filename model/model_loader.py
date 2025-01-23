import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,GenerationConfig
from model.huggingface import FineTuneConfig
from  model.SuffixManager import SuffixManager
from transformers import GPTJForCausalLM
from pathlib import Path


def load_base_model(
    model_path: str,
    device: str = 'cuda',
    use_half_precision: bool = True,
    **kwargs
) -> tuple:
    """
    加载基础模型和分词器
    
    Args:
        model_path: 模型路径
        device: 运行设备
        use_half_precision: 是否使用半精度(float16)
        **kwargs: 其他参数
    
    Returns:
        tuple: (model, tokenizer)
    """
    # 检查GPU可用性
    if device == 'cuda' and not torch.cuda.is_available():
        print("警告: 未检测到GPU,切换到CPU模式")
        device = 'cpu'
        use_half_precision = False
        
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
        torch_dtype=torch.float16 if use_half_precision else torch.float32,
        device_map='auto' if device=='cuda' else None
    )
    
    # 如果使用CPU,需要显式移动模型
    if device == 'cpu':
        model = model.to(device)
        
    return model, tokenizer

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
        # if 'vicuna-7b-v1.5' in model_path:
        #     tokenizer = AutoTokenizer.from_pretrained(
        #         "D:\ZLCODE\BabyLlama\models\\"+"vicuna-7b-distilled\\"+"final_model",
        #         trust_remote_code=True,
        #         padding_side='left'
        #     )
        
        # 设置填充标记
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if 'gpt-j-6b' in model_path:

            model = GPTJForCausalLM.from_pretrained(model_path,low_cpu_mem_usage = True,
            use_cache = False, torch_dtype=torch.float16).to(device)
        if 'final_model' in model_path:
            # 加载模型配置
            generation_config = GenerationConfig.from_pretrained(
                model_path,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                max_length=512,
                pad_token_id=tokenizer.pad_token_id
            )
            print("Generation config loaded.")

            # 加载完整模型（使用complete_model文件夹）
            model_path = Path(model_path) / "complete_model"
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                generation_config=generation_config
            )
            print("Model loaded successfully.")

        else:
            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device=='cuda' else torch.float32,
            device_map='auto' if device=='cuda' else None,
            low_cpu_mem_usage = True,
            use_cache = False
            )
        
        # 如果不需要微调,冻结模型参数
        if ft_config is None:
            for param in model.parameters():
                param.requires_grad = False
        
        # 将模型移动到指定设备
        if device == 'gpu':
            model = model.to(device)

        if param.requires_grad:    
            # 设置为评估模式
            model.train()
        else:
            # 设置为评估模式
            model.eval()
        
        # 创建后缀管理器
        suffix_manager = SuffixManager(
            tokenizer=tokenizer,
            use_system_instructions=use_system_instructions
        )
        if 'MindLLM-1b3' in model_path:
            tokenizer.bos_token_id = 50256
            tokenizer.eos_token_id = 50256
            tokenizer.pad_token_id = 50256
            tokenizer.padding_side = "left"
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        elif 'gpt-j-6b' in model_path:
            tokenizer.bos_token_id = 50256
            tokenizer.eos_token_id = 50256
            tokenizer.pad_token_id = 50256

            model.generation_config.pad_token_id = tokenizer.pad_token_id
        elif 'Llama-3.2-3B' in model_path:
            tokenizer.bos_token_id =  128000
            tokenizer.eos_token_id =  128001
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        print("Tokenizer vocab size:", tokenizer.vocab_size)
        print("Embedding matrix size:", model.get_input_embeddings().weight.size(0))
        print("成功加载proxy_model")
        return model, tokenizer
        
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        raise


def load_target_model(
    model_path: str,
    device: str = 'cuda',
    **kwargs
) -> tuple:
    """
    加载目标模型
    
    Args:
        model_path: 模型路径
        device: 运行设备
        **kwargs: 其他参数
    
    Returns:
        tuple: (model, tokenizer)
    """
    model, tokenizer = load_base_model(model_path, device, **kwargs)
    
    # 目标模型始终冻结且处于评估模式
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    return model, tokenizer
