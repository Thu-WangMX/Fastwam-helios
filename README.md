# FastWAM

Latent 域多数据集预训练框架。基于注册表系统，一个 YAML 配置文件驱动完整的预处理 + 训练流水线。

---

## 目录

1. [快速开始](#1-快速开始)
2. [数据格式](#2-数据格式)
3. [modality.json](#3-modalityjson数据准备第一步)
4. [注册新数据集](#4-注册新数据集dataconfig--mixture)
5. [配置文件](#5-配置文件)
6. [模型权重准备](#6-模型权重准备)
7. [预处理流水线](#7-预处理流水线)
8. [训练](#8-训练)
9. [AI Hub 集群提交](#9-ai-hub-集群提交)
10. [项目结构](#10-项目结构)

---

## 1. 快速开始

```bash
# 环境
export PYTHONPATH=src
PYTHON=/mnt/workspace/lintong.lt/env/fastwam/bin/python

# 1. 预处理（按顺序执行）
$PYTHON scripts/precompute_text_embeds.py   --config configs/robocasa365_pretrain.yaml
$PYTHON scripts/latent/scan_dataset_meta.py --config configs/robocasa365_pretrain.yaml --manifest_path ./data/manifest.jsonl
torchrun --nproc_per_node=8 scripts/latent/generate_latents.py --config configs/robocasa365_pretrain.yaml --manifest_path ./data/manifest.jsonl
$PYTHON scripts/compute_dataset_stats.py    --config configs/robocasa365_pretrain.yaml
$PYTHON scripts/split_train_val.py          --config configs/robocasa365_pretrain.yaml

# 2. 训练
torchrun --nproc_per_node=8 scripts/train_latent.py --config configs/robocasa365_pretrain.yaml
```

---

## 2. 数据格式

数据遵循 **LeRobot v2** 格式：

```
<dataset_dir>/
├── meta/
│   ├── info.json                    # 元信息（total_episodes, chunks_size, data_path 模板）
│   ├── tasks.jsonl                  # {"task_index": 0, "task": "Close the blender lid..."}
│   ├── modality.json                # state/action/video 的维度拆分定义
│   ├── episodes.jsonl               # episode 元信息
│   ├── stats_gr00t.json             # 归一化统计（compute_dataset_stats.py 生成）
│   ├── latent_sample_index_*.json   # sample 索引（scan_dataset_meta.py 生成）
│   └── train_val_split_*.json       # train/val 划分（split_train_val.py 生成）
├── data/chunk-000/                  # parquet（每行一帧：action, state, frame_index）
├── videos/chunk-000/                # mp4（按 camera 分目录）
├── latents/chunk-000/               # latent .pt（generate_latents.py 生成）
└── text_embeds/                     # text embedding .pt（precompute_text_embeds.py 生成）
```

**关键要求：**

- 每个 episode 的 parquet 行数必须与所有相机 mp4 帧数严格一致
- `frame_index` 列必须连续（从 0 递增，不允许跳帧）
- `modality.json` 必须定义 state/action/video/annotation 各子键的 `original_key`、`start`、`end`

---

## 3. modality.json（数据准备第一步）

每个子数据集的 `meta/modality.json` 定义了 parquet 列到逻辑键的映射关系。**这是数据准备的第一步**，所有预处理和训练都依赖它。

### 3.1 整体结构

```json
{
  "state": { ... },
  "action": { ... },
  "video": { ... },
  "annotation": { ... }
}
```

### 3.2 state / action 子键

```json
"action": {
  "end_effector_position": {
    "original_key": "action",
    "start": 5,
    "end": 8,
    "absolute": false,
    "rotation_type": null,
    "dtype": "float32"
  },
  "gripper_close": {
    "original_key": "action",
    "start": 11,
    "end": 12,
    "absolute": true,
    "rotation_type": null,
    "dtype": "float32"
  }
}
```

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `original_key` | parquet 中的真实列名 | None（默认用 `<modality>.<subkey>`） |
| `start` / `end` | 从该列向量中切片 `[start:end)` | 必填 |
| `absolute` | 越界 padding 策略：true=首尾帧补，false=0 补 | true |
| `rotation_type` | 旋转表示类型：euler/quat/axis_angle/rot6d/null | null |
| `dtype` | 数据类型 | "float32" |

### 3.3 video 子键

```json
"video": {
  "robot0_eye_in_hand": {
    "original_key": "observation.images.robot0_eye_in_hand"
  }
}
```

`original_key` 用于在 `info.json["features"]` 中查找分辨率/通道/fps，并拼接视频文件路径。

### 3.4 annotation 子键

```json
"annotation": {
  "human.task_description": {
    "original_key": "annotation.human.task_description"
  }
}
```

parquet 中存 `task_index`（int），加载时通过 `tasks.jsonl` 映射到真实文本。

### 3.5 完整示例（Robocasa365）

```json
{
  "state": {
    "base_position": {"original_key": "observation.state", "start": 0, "end": 3},
    "base_rotation": {"original_key": "observation.state", "start": 3, "end": 7},
    "end_effector_position_relative": {"original_key": "observation.state", "start": 7, "end": 10},
    "end_effector_rotation_relative": {"original_key": "observation.state", "start": 10, "end": 14},
    "gripper_qpos": {"original_key": "observation.state", "start": 14, "end": 16}
  },
  "action": {
    "base_motion": {"original_key": "action", "start": 0, "end": 4},
    "control_mode": {"original_key": "action", "start": 4, "end": 5},
    "end_effector_position": {"original_key": "action", "start": 5, "end": 8},
    "end_effector_rotation": {"original_key": "action", "start": 8, "end": 11},
    "gripper_close": {"original_key": "action", "start": 11, "end": 12}
  },
  "video": {
    "robot0_eye_in_hand": {"original_key": "observation.images.robot0_eye_in_hand"},
    "robot0_agentview_left": {"original_key": "observation.images.robot0_agentview_left"},
    "robot0_agentview_right": {"original_key": "observation.images.robot0_agentview_right"}
  },
  "annotation": {
    "human.task_description": {"original_key": "annotation.human.task_description"}
  }
}
```

### 3.6 与 DataConfig 的对应关系

modality.json 的子键名会被 DataConfig 的 `state_keys` / `action_keys` 引用：

```python
# modality.json 中定义: "end_effector_position": {"start": 5, "end": 8, ...}
# DataConfig 中引用:
action_keys = ["action.end_effector_position", ...]  # 前缀 "action." + 子键名
```

DataConfig 的 `transform()` 中的 `normalization_modes` 也按这个键名索引：

```python
normalization_modes = {"action.end_effector_position": "min_max", ...}
```

### 3.7 生成工具

使用 `/gen-modality` skill 可以自动从 parquet 推断并生成 modality.json。

同一 embodiment 的多个子数据集共享相同的 modality 结构，只需生成一份然后复制：

```bash
for dir in /path/to/datasets/*/; do
    cp modality_template.json "$dir/meta/modality.json"
done
```

---

## 4. 注册新数据集（DataConfig + Mixture）

准备好 modality.json（第 3 章）后，按以下步骤将新数据集接入训练。

### 4.1 编写 DataConfig

每个 embodiment 对应一个 `DataConfig` 类。文件：`src/fastwam/datasets/lerobot/registry/data_config.py`

使用 `/gen-dataconfig` skill 可自动生成，也可手动编写：

```python
class RobotwimDataConfig(BaseDataConfig):
    # ── 1. 模态键：引用 modality.json 中的子键名 ──
    embodiment_tag = EmbodimentTag.ARX5
    video_keys = [
        "video.high",              # modality.json → video.high
        "video.left_wrist",
        "video.right_wrist",
    ]
    state_keys = [
        "state.joint_position",    # modality.json → state.joint_position
        "state.gripper_qpos",
    ]
    action_keys = [
        "action.joint_position",   # modality.json → action.joint_position
        "action.gripper_close",
    ]
    language_keys = ["annotation.human.task_description"]

    # ── 2. 时间窗口 ──
    observation_indices = [0]          # 当前帧
    action_indices = list(range(32))   # 未来 32 步 action chunk

    # ── 3. 视频拼接 ──
    video_concat_mode = "top_bottom"
    video_size = (384, 320)            # 拼接后一帧的 (H, W)，用于 bucket 分组

    # ── 4. 统计方式 ──
    action_mode = "abs"                # "abs" / "delta" / "rel"

    # ── 5. Text prompt 模板 ──
    prompt_embodiment = "A dual-arm ARX5 robot"
    prompt_views = "observed from 3 cameras (top-down, left wrist, right wrist)"
    prompt_control = "controlled by joint position commands"

    # ── 6. modality_config（框架必需） ──
    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    # ── 7. 归一化链（引用 modality.json 的子键名） ──
    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.action_keys + self.state_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.joint_position": "min_max",
                    "action.gripper_close": "binary",
                },
            ),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.joint_position": "min_max",
                    "state.gripper_qpos": "min_max",
                },
            ),
        ])

    # ── 8. 视频处理流水线 ──
    def video_transforms(self) -> list:
        return [
            PerCameraResize(sizes={
                "high": (256, 320),
                "left_wrist": (128, 160),
                "right_wrist": (128, 160),
            }),
            ConcatCameras(camera_keys=self._camera_keys_stripped(), mode="top_bottom"),
            Normalize(),
        ]
```

#### video_transforms 示例

| 场景 | 流水线 | 输出尺寸 |
|------|--------|----------|
| 同分辨率水平拼接 | `ConcatCameras(horizontal)` → `Normalize` | (256, 768) |
| 不同分辨率上下拼接 | `PerCameraResize` → `ConcatCameras(top_bottom)` → `Normalize` | (384, 320) |
| 单相机裁剪缩放 | `PerCameraCenterCrop` → `ConcatCameras` → `Resize` → `CenterCrop` → `Normalize` | (224, 224) |
| pad 统一后拼接 | `ConcatCameras(pad_to=...)` → `Resize` → `CenterCrop` → `Normalize` | (256, 768) |

视频 transform 积木（三层）：

| 层 | 类 | 输入 → 输出 |
|---|---|---|
| 单视角 | `PerCameraResize`, `PerCameraCenterCrop` | dict → dict |
| 拼接 | `ConcatCameras(mode, pad_to)` | dict → Tensor |
| 整体 | `Resize`, `CenterCrop`, `Pad`, `Normalize` | Tensor → Tensor |

### 4.2 注册 embodiment

在 `ROBOT_TYPE_CONFIG_MAP` 中注册（`data_config.py` 底部）：

```python
ROBOT_TYPE_CONFIG_MAP: dict[str, BaseDataConfig] = {
    "robocasa365": Robocasa365DataConfig(),
    "robotwin": RobotwimDataConfig(),       # 新增
}
```

如需新的 `EmbodimentTag`，在 `embodiment_tags.py` 中添加：

```python
class EmbodimentTag(Enum):
    FRANKA = "franka"
    ARX5 = "arx5"       # 新增
```

### 4.3 编写 Mixture

在 `mixtures.py` 中定义数据集混合：

```python
DATASET_NAMED_MIXTURES = {
    "robotwin_all": [
        ("/path/to/robotwin/task_A", 1.0, "robotwin"),
        ("/path/to/robotwin/task_B", 1.0, "robotwin"),
        ("/path/to/robotwin/task_C", 2.0, "robotwin"),  # 权重 2.0 = 采样概率翻倍
    ],

    # 跨 embodiment 混合训练
    "pretrain_mixed": [
        ("/path/to/robocasa365/task_1", 1.0, "robocasa365"),
        ("/path/to/robocasa365/task_2", 1.0, "robocasa365"),
        ("/path/to/robotwin/task_A", 1.0, "robotwin"),     # 不同 robot_type
    ],
}
```

**跨 embodiment 混合说明：**

- 同一 `robot_type` 的所有子数据集共享同一个 `DataConfig`，归一化统计会在训练时按权重合并
- 不同 `robot_type` 的数据集会按各自的 `video_size` 分到不同的 bucket，同 bucket 内组 batch，不同 bucket 不混合
- `weight` 控制采样概率：`prob_i = weight_i * num_samples_i / sum(weight_j * num_samples_j)`

### 4.4 完整流程

```
1. /gen-modality     → 每个子数据集生成 meta/modality.json
2. /gen-dataconfig   → 生成 DataConfig 类 + 注册到 ROBOT_TYPE_CONFIG_MAP
3. mixtures.py       → 添加 mixture 定义
4. configs/xxx.yaml  → 新建配置文件
5. 预处理 + 训练      → 按后续章节执行
```

---

## 5. 配置文件

一个 YAML 文件包含所有配置：

```yaml
# configs/robocasa365_pretrain.yaml

data:
  data_mix: robocasa365_pretrain_all     # 注册表中的 mixture 名
  action_dim: 12
  proprio_dim: 16
  stride: 1
  action_video_freq_ratio: 4
  num_frames: 33
  val_set_proportion: 0.01
  base_seed: 42
  zero_pad_len: 18

model:
  model_id: Wan-AI/Wan2.2-TI2V-5B
  tokenizer_model_id: Wan-AI/Wan2.1-T2V-1.3B
  tokenizer_max_len: 128
  multi_term_memory: true
  mtm_history_sizes: [16, 2, 1]
  mtm_pred_size: 2
  # ... 完整模型配置见文件

trainer:
  output_dir: ./runs/robocasa365_pretrain
  batch_size: 16
  num_workers: 12
  learning_rate: 1.0e-4
  lr_scheduler_type: cosine
  num_epochs: 10
  max_steps: 10000
  # ... 完整训练配置见文件

wandb:
  enabled: false
  project: fast-wam-latent
  name: robocasa365_pretrain
```

---

## 6. 模型权重准备

### 6.1 ActionDiT 权重预处理

ActionDiT 是独立的 action 预测网络（hidden_dim=1024），需要从 VideoDiT（hidden_dim=3072）权重中线性插值初始化：

```bash
bash actiondit.sh
```

产物路径需要与 yaml 中 `model.action_dit_pretrained_path` 对应。只需运行一次，后续复用同一份权重。

---

## 7. 预处理流水线

所有脚本通过 `--config` 指定 YAML 配置，从 `data.data_mix` 自动解析数据集。

### 7.0 Stride 推荐（Latent 生成前必做）

stride 决定了从每个 episode 中滑窗切片的步长，直接影响 manifest 大小和训练数据多样性。**在生成 latent 之前确定 stride，写入 yaml，后续步骤都会读取。**

```bash
python scripts/latent/recommend_stride.py \
    --config configs/robocasa365_pretrain.yaml \
    --bs 64 --steps 60000

# OSS 模式
python scripts/latent/recommend_stride.py \
    --config configs/robocasa365_pretrain.yaml \
    --bs 64 --steps 60000 \
    --backend oss --oss_mount /data/oss_bucket_0:oss://xlab-dev
```

输出示例（robocasa365_pretrain_all，32146 episodes，2918 万帧）：

```
=== Stride 推荐 ===
  data_mix          = robocasa365_pretrain_all
  num_frames        = 33
  episodes          = 32,146  (total frames = 29.18M)
  batch_size        = 64
  steps             = 60,000
  budget (bs*steps) = 3.84M samples
  target_epochs     = 1.0

  stride    #records     #chunks  budget%   epochs  fits
  ------ ----------- ----------- -------- -------- -----
       1       1.06M      28.15M   733.1%    0.14x     N
       3      353.5K       9.39M   244.6%    0.41x     N
      11       96.4K       2.57M    67.0%    1.49x     Y  <- recommended
      33       32.1K      868.6K    22.6%    4.42x     Y

  推荐 stride = 11  (= num_frames / 3, 即每 episode 3 个 offset)
    manifest 大小  = 96.4K records / 2.57M chunks
    预算消耗       = 67.0%   =>   1.49x epochs
```

确定后写入 yaml：

```yaml
data:
  stride: 11    # 根据推荐结果设置
```

**stride 参数说明：**

| stride | offset 数 | 数据量 | 适用场景 |
|--------|----------|--------|----------|
| `33`（= num_frames） | 1 | 1x | 快速实验，最少存储 |
| `11` | 3 | 3x | 推荐：多样性与开销平衡 |
| `1` | 33 | 33x | 最大多样性，存储开销 33 倍 |

### 7.1 Text Embedding 生成

为每个子数据集的 task 描述生成 T5 embedding，存到 `<dataset_dir>/text_embeds/`。

```bash
# 单卡
python scripts/precompute_text_embeds.py --config configs/robocasa365_pretrain.yaml

# 多卡并行
torchrun --nproc_per_node=8 scripts/precompute_text_embeds.py --config configs/robocasa365_pretrain.yaml
```

Prompt 由 DataConfig 的三部分模板 + task 描述组成：

```
"{prompt_embodiment}, {prompt_views}, {prompt_control}. {task}"
```

### 7.2 Latent 生成（两步）

**Stage-1：扫描元数据 → manifest**

```bash
# NAS 挂载
python scripts/latent/scan_dataset_meta.py \
    --config configs/robocasa365_pretrain.yaml \
    --manifest_path ./data/manifest.jsonl \
    --num_workers 32

# OSS（无挂载）
python scripts/latent/scan_dataset_meta.py \
    --config configs/robocasa365_pretrain.yaml \
    --manifest_path oss://bucket/.../manifest.jsonl \
    --backend oss --oss_mount /data/oss_bucket_0:oss://xlab-dev
```

产物：

- `manifest.jsonl` — 每行一个 episode/offset 的 chunk 信息
- `<dataset_dir>/meta/latent_sample_index_stride{S}_nf{N}.json` — sample 索引

**Stage-2：分布式 VAE 编码**

```bash
torchrun --nproc_per_node=8 scripts/latent/generate_latents.py \
    --config configs/robocasa365_pretrain.yaml \
    --manifest_path ./data/manifest.jsonl
```

产物：`<dataset_dir>/latents/chunk-XXX/episode_YYYYYY_offset_ZZ.pt`

### 7.3 归一化统计

按 DataConfig 的 `action_mode`（abs/delta/rel）计算 action/state 统计。

```bash
python scripts/compute_dataset_stats.py \
    --config configs/robocasa365_pretrain.yaml \
    --num_workers 32
```

产物：`<dataset_dir>/meta/stats_gr00t.json`（含缓存校验，action_mode 不匹配自动重算）

### 7.4 Train/Val 划分

按轨迹级别切分，`val = max(1, round(val_set_proportion * total_episodes))`。

```bash
python scripts/split_train_val.py --config configs/robocasa365_pretrain.yaml
```

产物：`<dataset_dir>/meta/train_val_split_stride{S}_nf{N}.json`

---

## 8. 训练

```bash
# 单卡
python scripts/train_latent.py --config configs/robocasa365_pretrain.yaml

# 多卡
torchrun --standalone --nproc_per_node=8 scripts/train_latent.py \
    --config configs/robocasa365_pretrain.yaml
```

训练器自动：

- 从注册表解析 `data_mix` → 构建 `LatentMixtureDataset`（多数据集加权混合）
- 按 `video_size` 分 bucket 组 batch（不同分辨率不混合）
- 按 `train_val_split` 划分 train/val
- 支持 `resume`（在 `trainer.resume` 填 checkpoint 路径）

### 产物

```
<output_dir>/
├── config.yaml              # 保存的完整配置
├── train_metrics.jsonl       # 逐步训练指标
├── checkpoints/
│   ├── weights/step_XXXXXX.pt
│   └── state/step_XXXXXX/
└── eval/                    # 验证视频
```

---

## 9. AI Hub 集群提交

预处理和训练均可通过 AI Hub 提交到集群执行。

### 9.1 预处理流水线

按顺序执行：

```bash
# Step 1: Text Embedding（AI Hub 多卡，需 GPU）
bash scripts_aihub/prepare/aihub_precompute_text_embeds.sh

# Step 2: 本地预处理（scan + stats + split，CPU，无需 GPU）
bash scripts_aihub/prepare/local_prepare.sh

# Step 3: Latent 编码（AI Hub 多卡，需 GPU，推荐 128 卡）
bash scripts_aihub/prepare/aihub_generate_latents.sh
```

### 9.2 训练

```bash
bash scripts_aihub/train/aihub_train_latent.sh
```

### 9.3 脚本配置

每个脚本顶部有需要修改的变量：

| 变量 | 说明 | 示例 |
|------|------|------|
| `CONFIG` | task yaml 路径 | `configs/robocasa365_pretrain.yaml` |
| `NUM_WORKERS` / `NUM_GPUS` | 并行 GPU 数 | `32` / `128` |
| `MODEL_BASE_PATH` | 预训权重路径 | `/mnt/workspace/.../pretrain_checkpoint` |
| `MANIFEST_PATH` | manifest JSONL 路径 | `./data/robocasa365_pretrain.jsonl` |

AI Hub 的队列、namespace、token 等认证信息需要在脚本中填写。


---

## 10. 项目结构

```
FastWAM/
├── configs/
│   └── robocasa365_pretrain.yaml          # 完整配置（data + model + trainer + wandb）
├── scripts_aihub/
│   ├── prepare/
│   │   ├── aihub_precompute_text_embeds.sh   # AI Hub: text embedding
│   │   ├── aihub_generate_latents.sh         # AI Hub: latent 编码
│   │   └── local_prepare.sh                  # 本地: scan + stats + split
│   └── train/
│       └── aihub_train_latent.sh
├── scripts/
│   ├── precompute_text_embeds.py          # Step 1: text embedding
│   ├── latent/
│   │   ├── scan_dataset_meta.py           # Step 2a: 扫描元数据 → manifest
│   │   └── generate_latents.py            # Step 2b: 分布式 VAE 编码
│   ├── compute_dataset_stats.py           # Step 3: 归一化统计
│   ├── split_train_val.py                 # Step 4: train/val 划分
│   └── train_latent.py                    # Step 5: 训练入口
├── src/fastwam/
│   ├── datasets/lerobot/
│   │   ├── registry/                      # 注册表系统
│   │   │   ├── data_config.py             #   DataConfig 定义
│   │   │   ├── mixtures.py                #   数据集混合定义
│   │   │   ├── embodiment_tags.py         #   EmbodimentTag 枚举
│   │   │   ├── prompt_utils.py            #   prompt 构造
│   │   │   ├── stats_utils.py             #   stats 计算/缓存
│   │   │   ├── schema.py                  #   pydantic 模型
│   │   │   └── transform/                 #   变换积木
│   │   │       ├── video.py               #     7 个视频原子 transform
│   │   │       ├── state_action.py        #     action/state 归一化
│   │   │       ├── base.py                #     基类
│   │   │       └── concat.py              #     concat 变换
│   │   └── latents/                       # 数据集类
│   │       ├── latent_single_dataset.py   #   单数据集加载器
│   │       ├── latent_mixture_dataset.py  #   多数据集混合
│   │       ├── mixture_bucket_sampler.py  #   bucket 分组 sampler
│   │       ├── latent_io.py               #   IO 工具
│   │       └── latent_bucket_sampler.py   #   manifest sampler
│   ├── models/wan22/                      # 模型（FastWAMMemory 等）
│   ├── inference/                         # 推理
│   ├── trainer_latent.py                  # Wan22LatentTrainer
│   └── utils/                             # 工具
└── .claude/commands/
    └── gen-dataconfig.md                  # /gen-dataconfig skill
```

