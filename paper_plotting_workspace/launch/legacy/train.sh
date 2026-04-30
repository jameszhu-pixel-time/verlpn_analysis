# python train_strategy_np_driver.py \
#   --train_base /DATA/disk2/zhurui/A_entry/results/verl_2_3/2.jsonl \
#   --train_post ref=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/annealed/verl_rollouts/3.jsonl \
#   --train_post n1=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_1__3.jsonl \
#   --train_post n2=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_2__3.jsonl \
#   --train_post n3=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_3__3.jsonl \
#   --train_post 3=/DATA/disk2/zhurui/A_entry/results/verl_2_3/3.jsonl  \
#   --train_post annealed=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/annealed/verl_rollouts/3.jsonl\
#   --train_ref ref \
#   --drivers_py /DATA/disk2/zhurui/A_entry/results/verl_driver_new2/drivers.py \
#   --outdir /DATA/disk2/zhurui/A_entry/results/train_np_driver_out \
#   --x_metric d_skew \
#   --subset ALL4 \
#   --max_len 3072 \
#   --pad_id 151643 \
#   --bridge

# python train_strategy_np_driver.py \
#   --train_base /DATA/disk2/zhurui/A_entry/ablation_train/dapo/verl_rollouts/3.jsonl \
#   --train_post ref=/DATA/disk2/zhurui/A_entry/ablation_train/dapo/verl_rollouts/3.jsonl \
#   --train_post n1=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_1__3.jsonl \
#   --train_post n2=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_2__3.jsonl \
#   --train_post n3=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_3__3.jsonl \
#   --train_post annealed=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/annealed/verl_rollouts/3.jsonl \
#   --train_ref n1 \
#   --val_base /DATA/disk2/zhurui/A_entry/ablation_train/dapo/verl_rollouts/3.jsonl \
#   --val_post n1=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_1/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl  \
#   --val_post n2=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_2/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl  \
#   --val_post n3=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_3/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl   \
#   --val_post ref=/DATA/disk2/zhurui/A_entry/ablation_train/dapo/vllm_rollouts_train/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl    \
#   --val_post annealed=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/annealed/vllm_rollouts_train/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
#   --drivers_py /DATA/disk2/zhurui/A_entry/results/verl_driver_new2/drivers.py \
#   --outdir /DATA/disk2/zhurui/A_entry/results/train_np_driver_out_n1 \
#   --x_metric tail_mass \
#   --subset ALL4 \
#   --max_len 3072 \
#   --pad_id 151643 \
#   --bridge

python train_rollout_driver_with_nplabels.py \
  --train ref=/DATA/disk2/zhurui/A_entry/ablation_train/dapo/verl_rollouts/3.jsonl \
  --train n1=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_1__3.jsonl \
  --train n2=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_2__3.jsonl \
  --train n3=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_3__3.jsonl \
  --train annealed=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/annealed/verl_rollouts/3.jsonl \
  --train_ref n1 \
  --np_base /DATA/disk2/zhurui/A_entry/ablation_train/dapo/verl_rollouts/3.jsonl \
  --np_post n1=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_1/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --np_post n2=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_2/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --np_post n3=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_3/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --np_post ref=/DATA/disk2/zhurui/A_entry/ablation_train/dapo/vllm_rollouts_train/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --np_post annealed=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/annealed/vllm_rollouts_train/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --drivers_py /DATA/disk2/zhurui/A_entry/Inference_pipeline/driver-NP/drivers.py \
  --outdir /DATA/disk2/zhurui/A_entry/results/train_driver_np_labels_out_new \
  --subset ALL4 \
  --x_metric d_tail_mass \
  --max_len 3072 \
  --pad_id 151643 \
  --bridge