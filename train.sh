conda activate count
export HF_DATASETS_CACHE="/NAS/wujunkang/guizhiyu/cache"
export HF_HOME="/NAS/wujunkang/guizhiyu/cache"
export HUGGINGFACE_HUB_CACHE="/NAS/wujunkang/guizhiyu/cache"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=1,5
accelerate launch train.py --config_file ./config/zero2cfg.json --model_config_path ./config/70m_offload.json