# LeRobot HIL-SERL 仿真训练说明

本文档用于说明如何基于 LeRobot 的 HIL-SERL 仿真环境进行数据采集、训练、继续训练以及常见问题排查。

## 参考资料

建议先阅读 Hugging Face 官方教程：

- https://hugging-face.cn/docs/lerobot/hilserl_sim

在开始前，请提前配置好：

- Weights & Biases（wandb）
- Hugging Face 账号
- Hugging Face 数据集仓库 / 模型仓库

后续可以在配置文件中将相关字段修改为自己的仓库地址和实验配置，例如：

- `src/gym_hil_env.json`
- `src/train_gym_hil.json`

## 环境配置

### 1. 创建 Conda 环境

```bash
conda create -n lerobot python=3.12
conda activate lerobot
```

### 2. 安装依赖

在 LeRobot 项目根目录下执行：

```bash
pip install -e ".[hilserl]"
```

该算法在mujoco环境下运行，该环境包含在gym_hil包中，为提供pick_place场景，需将gym_hil包中的scene.xml文件替换为下面路径的xml文件：

```bash
/lerobot/src/scene.xml
```


## 数据采集与环境交互

离线数据已经采集完成，并保存到了 Hugging Face 仓库中。相关配置可参考：

```text
src/gym_hil_env.json
```

如果需要重新采集数据并训练，可以修改配置文件后运行：

```bash
python -m lerobot.rl.gym_manipulator --config_path /home/用户名/lerobot/src/gym_hil_env.json
```


如果只是想熟悉键盘控制，不重新采集数据，可以将配置文件中的 `mode` 字段改为：

```json
"mode": null
```

然后再次运行：

```bash
python -m lerobot.rl.gym_manipulator --config_path src/gym_hil_env.json
```


## 从已有权重继续训练

在 `src/outputs` 路径下已经保存了训练 8000 步后的权重，可以直接基于该 checkpoint 继续训练。

需要启动两个进程。

### 终端 1：启动 Learner

如果 checkpoint 路径相对于当前项目根目录是 `src/outputs/...`，运行：

```bash
python -m lerobot.rl.learner \
  --config_path=src/outputs/train/2026-06-03/14-33-57_0603/checkpoints/008000/pretrained_model/train_config.json \
  --resume=true
```

### 终端 2：启动 Actor

```bash
python -m lerobot.rl.actor \
  --config_path=src/outputs/train/2026-06-03/14-33-57_0603/checkpoints/008000/pretrained_model/train_config.json \
  --resume=true
```



