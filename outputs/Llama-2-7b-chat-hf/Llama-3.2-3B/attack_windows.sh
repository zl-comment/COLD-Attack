python cold_decoding.py `
    --seed 12 `
    --mode "proxy" `
    --pretrained_model "Llama-2-7b-chat-hf" `
    --init-temp 1 `
    --length 20 `
    --max-length 20 `
    --num-iters 2000 `
    --min-iters 0 `
    --goal-weight 0.001 `
    --rej-weight 0.001 `
    --stepsize 1e-10 `
    --kl_max_weight 0.1 `
    --noise-iters 1 `
    --win-anneal-iters 1000 `
    --start 0 `
    --end 50 `
    --lr-nll-portion 1.0 `
    --topk 10 `
    --output-lgt-temp 1 `
    --verbose `
    --straight-through `
    --large-noise-iters 50,200,500,1500 `
    --large_gs_std 0.1,0.05,0.01,0.001 `
    --stepsize-ratio 1 `
    --batch-size 8 `
    --print-every 1000 `
    --proxy_model "Llama-3.2-3B" `
    --proxy_model_path "D:\ZLCODE\model\Llama-3.2-3B" `
    --wandb_project COLD_Attack_proxy_Llama-3.2-3B `
     --wandb