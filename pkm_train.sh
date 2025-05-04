conda activate count
export HF_DATASETS_CACHE="/NAS/wujunkang/guizhiyu/cache"
export HF_HOME="/NAS/wujunkang/guizhiyu/cache"
export HUGGINGFACE_HUB_CACHE="/NAS/wujunkang/guizhiyu/cache"
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1
accelerate launch --config_file ./config/zero2.json pkm_train.py --model_config_path ./config/pkm160m.json