conda activate gemma3n
export HF_DATASETS_CACHE="/NAS/wujunkang/guizhiyu/cache"
export HF_HOME="/NAS/wujunkang/guizhiyu/cache"
export HUGGINGFACE_HUB_CACHE="/NAS/wujunkang/guizhiyu/cache"
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=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
export WANDB_BASE_URL=https://api.wandb-cn.top
export CUDA_VISIBLE_DEVICES=6,7
accelerate launch --config_file ./config/zero2.json ./mole/gemma3n/train_gemma3n.py --model_config_path ./mole/gemma3n/config/debug.yaml