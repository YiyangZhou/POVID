# conda env 
source activate /data/yiyang_zhou/miniconda3/envs/llava
cd /data/yiyang_zhou/workplace/LLaVA/
deepspeed llava/train/train_dpo_inherent.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /data/yiyang_zhou/workplace/LLaVA/checkpoint/output/POVID_stage_one_merged \
    --version v1 \
    --data_path /data/yiyang_zhou/workplace/datashop/dpo_data/POVID_preference_data_for_VLLMs.json \
    --image_folder /data/yiyang_zhou/workplace/datashop/dpo_data/coco \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /data/yiyang_zhou/workplace/LLaVA/checkpoint/output/POVID_stage_two_LoRa \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1\
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 1e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to wandb
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \