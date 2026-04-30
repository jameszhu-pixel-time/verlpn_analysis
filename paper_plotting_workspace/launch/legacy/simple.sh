python simple.py \
  --train ref=/DATA/disk2/zhurui/A_entry/ablation_train/dapo/verl_rollouts/3.jsonl \
  --train n1=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_1__3.jsonl \
  --train n2=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_2__3.jsonl \
  --train n3=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_3__3.jsonl \
  --train annealed=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/annealed/verl_rollouts/3.jsonl \
  --np_base /DATA/disk2/zhurui/A_entry/ablation_train/dapo/verl_rollouts/3.jsonl \
  --np_post n1=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_1/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --np_post n2=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_2/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --np_post n3=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/normal_3/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --np_post ref=/DATA/disk2/zhurui/A_entry/ablation_train/dapo/vllm_rollouts_train/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --np_post annealed=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/annealed/vllm_rollouts_train/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --drivers_py //DATA/disk2/zhurui/A_entry/Inference_pipeline/driver-NP/drivers.py \
  --outdir /DATA/disk2/zhurui/A_entry/results/intra_np_driver_out1 \
  --subset ALL4 \
  --max_len 3072 \
  --pad_id 151643 \
  --bins 60


python simple_essay.py \
  --train ref=/DATA/disk2/zhurui/A_entry/ablation_train/dapo/verl_rollouts/3.jsonl \
  --train n2=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_2__3.jsonl \
  --train n3=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_3__3.jsonl \
  --train annealed=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/annealed/verl_rollouts/3.jsonl \
  --np_base ref=/DATA/disk2/zhurui/A_entry/ablation_train/dapo/verl_rollouts/3.jsonl \
  --np_base n2=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_2__3.jsonl \
  --np_base n3=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_3__3.jsonl \
  --np_base annealed=/DATA/disk2/zhurui/A_entry/ablation_train/ablation_study_step_2_3/annealed/verl_rollouts/3.jsonl \
  --np_post n2=/DATA/disk1/zhurui/ablation_study_step_2_3/normal_2/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --np_post n3=/DATA/disk1/zhurui/ablation_study_step_2_3/normal_3/vllm_rollouts_training/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --np_post ref=/DATA/disk2/zhurui/A_entry/ablation_train/dapo/vllm_rollouts_train/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --np_post annealed=/DATA/disk1/zhurui/ablation_study_step_2_3/annealed/vllm_rollouts_train/grpo_step3-dapo17k-8_rollout-test-temp_1.0-top_p_1.0-top_k_-1.jsonl \
  --drivers_py //DATA/disk2/zhurui/A_entry/Inference_pipeline/driver-NP/drivers.py \
  --outdir /DATA/disk2/zhurui/A_entry/results/paper_codex/plot2 \
  --subset all \
  --max_len 3072 \
  --pad_id 151643 \
  --bins 60


python compare_two_driver_distributions.py \
  --a ref=/DATA/disk2/zhurui/A_entry/ablation_train/dapo/verl_rollouts/3.jsonl \
  --b n1=/DATA/disk2/zhurui/A_entry/results/verl_2_3/normal_1__3.jsonl \
  --drivers_py /DATA/disk2/zhurui/A_entry/Inference_pipeline/driver-NP/drivers.py \
  --outdir /DATA/disk2/zhurui/A_entry/results/driver_compare_out \
  --max_len 3072 \
  --pad_id 151643
