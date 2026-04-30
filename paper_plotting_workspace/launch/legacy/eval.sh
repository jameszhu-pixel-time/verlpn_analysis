# python eval_strategy_np_driver.py \
#   --train ref=/DATA/disk2/zhurui/A_entry/results/verl_2_3/3.jsonl \
#   --train n1=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_1__3.jsonl \
#   --train n2=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_2__3.jsonl \
#   --train n3=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_3__3.jsonl \
#   --train annealed=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/annealed/verl_rollouts/3.jsonl \
#   --train_ref ref \
#   --val_base /DATA/disk2/zhurui/A_entry/results/amc_2_3/grpo_step2-amc23-64_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
#   --val_post n1=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_1/vllm_rollouts/grpo_step3-amc23-64_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
#   --val_post n2=/DATA/disk2/zhurui/A_entry/results/amc_2_3/normal_2_step3-amc23-64_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
#   --val_post n3=/DATA/disk2/zhurui/A_entry/results/amc_2_3/normal_3_step3-amc23-64_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
#   --val_post annealed=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/annealed/vllm_rollouts/annealed_step3-amc23-64_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
#   --drivers_py /DATA/disk2/zhurui/A_entry/results/verl_driver_new2/drivers.py \
#   --outdir /DATA/disk2/zhurui/A_entry/results/eval_strategy_np_driver_out \
#   --x_metric d_skew \
#   --subset ALL4 \
#   --max_len 3072 \
#   --pad_id 151643


#aime

python eval_strategy_np_driver.py \
  --train ref=/DATA/disk2/zhurui/A_entry/results/verl_2_3/3.jsonl \
  --train n1=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_1__3.jsonl \
  --train n2=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_2__3.jsonl \
  --train n3=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_3__3.jsonl \
  --train_ref n1 \
  --val_base /DATA/disk2/zhurui/A_entry/results/aime_2_3/grpo_step2-aime2025-64_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --val_post n1=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_1/vllm_rollouts2/grpo_step3-aime2025-64_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --val_post n2=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_2/vllm_rollouts2/grpo_step3-aime2025-64_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --val_post n3=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_3/vllm_rollouts2/grpo_step3-aime2025-64_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --val_post ref=/DATA/disk2/zhurui/A_entry/ablation_train/dapo/vllm_rollouts2/grpo_step3-aime2025-64_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --drivers_py /DATA/disk2/zhurui/A_entry/results/verl_driver_new2/drivers.py \
  --outdir /DATA/disk2/zhurui/A_entry/results/eval_strategy_np_driver_out_aime3 \
  --x_metric skew \
  --subset ALL4 \
  --max_len 3072 \
  --pad_id 151643
#beyond

python eval_strategy_np_driver.py \
  --train ref=/DATA/disk2/zhurui/A_entry/results/verl_2_3/3.jsonl \
  --train n1=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_1__3.jsonl \
  --train n2=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_2__3.jsonl \
  --train n3=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_3__3.jsonl \
  --train annealed=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/annealed/verl_rollouts/3.jsonl \
  --train_ref annealed \
  --val_base /DATA/disk2/zhurui/A_entry/results/beyond_2_3/grpo_step2-beyond-64_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --val_post n1=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_1/vllm_rollouts3/grpo_step3-beyond-64_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --val_post n2=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_2/vllm_rollouts3/grpo_step3-beyond-64_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --val_post n3=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_3/vllm_rollouts3/grpo_step3-beyond-64_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --val_post annealed=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/annealed/vllm_rollouts/annealed_step3-beyond-64_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --val_post ref=/DATA/disk2/zhurui/A_entry/ablation_train/dapo/vllm_rollouts2/grpo_step3-beyond-64_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --drivers_py /DATA/disk2/zhurui/A_entry/results/verl_driver_new2/drivers.py \
  --outdir /DATA/disk2/zhurui/A_entry/results/eval_strategy_np_driver_out_beyond3 \
  --x_metric d_skew \
  --subset ALL4 \
  --max_len 3072 \
  --pad_id 151643
#amc
python eval_strategy_np_driver.py \
  --train ref=/DATA/disk2/zhurui/A_entry/results/verl_2_3/3.jsonl \
  --train n1=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_1__3.jsonl \
  --train n2=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_2__3.jsonl \
  --train n3=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_3__3.jsonl \
  --train annealed=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/annealed/verl_rollouts/3.jsonl \
  --train_ref annealed \
  --val_base /DATA/disk2/zhurui/A_entry/results/amc_2_3/grpo_step2-amc23-64_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --val_post n1=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_1/vllm_rollouts3/grpo_step3-amc23-64_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --val_post n2=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_2/vllm_rollouts3/grpo_step3-amc23-64_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --val_post n3=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_3/vllm_rollouts3/grpo_step3-amc23-64_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --val_post annealed=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/annealed/vllm_rollouts/annealed_step3-amc23-64_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --val_post ref=/DATA/disk2/zhurui/A_entry/ablation_train/dapo/vllm_rollouts2/grpo_step3-amc23-64_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --drivers_py /DATA/disk2/zhurui/A_entry/results/verl_driver_new2/drivers.py \
  --outdir /DATA/disk2/zhurui/A_entry/results/eval_strategy_np_driver_out_amc1 \
  --x_metric tail_mass \
  --subset ALL4 \
  --max_len 3072 \
  --pad_id 151643
#train

python eval_strategy_np_driver.py   \
  --train ref=/DATA/disk2/zhurui/A_entry/ablation_train/dapo/verl_rollouts/3.jsonl \
  --train n1=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_1__3.jsonl  \
  --train n2=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_2__3.jsonl   \
  --train n3=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_3__3.jsonl   \
  --train annealed=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/annealed/verl_rollouts/3.jsonl   \
  --train_ref annealed   \
  --val_base /DATA/disk2/zhurui/A_entry/ablation_train/dapo/verl_rollouts/3.jsonl \
  --val_post n1=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_1/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl  \
  --val_post n2=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_2/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl  \
  --val_post n3=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_3/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl   \
  --val_post ref=/DATA/disk2/zhurui/A_entry/ablation_train/dapo/vllm_rollouts_train/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl    \
  --val_post annealed=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/annealed/vllm_rollouts_train/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --drivers_py /DATA/disk2/zhurui/A_entry/results/verl_driver_new2/drivers.py \
  --outdir /DATA/disk2/zhurui/A_entry/results/train_np_driver_out1  \
  --x_metric tail_mass \
  --subset ALL4  \
  --max_len 3072 \
  --pad_id 151643 
  