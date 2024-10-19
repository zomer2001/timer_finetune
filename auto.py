import subprocess

# 配置参数
model_name = "Timer"
seq_len = 672
label_len = 576
pred_len = 96
output_len = 96
patch_len = 96
ckpt_path = "checkpoints/Timer_forecast_1.0.ckpt"
data = "electricity"
subset_rand_ratios = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1]

# 遍历不同的数据稀缺比例并运行命令
for subset_rand_ratio in subset_rand_ratios:
    command = [
        "torchrun", "--nnodes=1", "--nproc_per_node=4", "run.py",
        "--task_name", "forecast",
        "--is_training", "0",
        "--ckpt_path", ckpt_path,
        "--root_path", f"./dataset/{data}/",
        "--data_path", f"{data}.csv",
        "--data", "custom",
        "--model_id", f"electricity_sr_{subset_rand_ratio}",
        "--model", model_name,
        "--features", "M",
        "--seq_len", str(seq_len),
        "--label_len", str(label_len),
        "--pred_len", str(pred_len),
        "--output_len", str(output_len),
        "--e_layers", "8",
        "--factor", "3",
        "--des", "Exp",
        "--d_model", "1024",
        "--d_ff", "2048",
        "--batch_size", "2048",
        "--learning_rate", "3e-5",
        "--num_workers", "4",
        "--patch_len", str(patch_len),
        "--train_test", "0",
        "--subset_rand_ratio", str(subset_rand_ratio),
        "--itr", "1",
        "--gpu", "0",
        "--use_ims",
        "--use_multi_gpu"
    ]

    # 打印命令（可选）
    print(f"Running: {' '.join(command)}")

    # 执行命令
    subprocess.run(command)
