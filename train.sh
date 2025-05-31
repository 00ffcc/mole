conda activate py12cu118
export HF_DATASETS_CACHE="/NAS/wujunkang/guizhiyu/cache"
export HF_HOME="/NAS/wujunkang/guizhiyu/cache"
export HUGGINGFACE_HUB_CACHE="/NAS/wujunkang/guizhiyu/cache"
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1
accelerate launch --config_file ./config/zero2.json trainer/ple_train.py --model_config_path ../config/ple14m.yaml