# Wan2.2 双节点部署指南

## 环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Linux (Ubuntu 20.04+) |
| GPU | 每台服务器 4×A100 40GB |
| Docker | >= 24.0 |
| Docker Compose | v2 (插件模式) |
| NVIDIA Container Toolkit | 已安装并配置 |
| 网络 | 两台服务器在同一局域网，可互相访问 TCP 29500 和 6379 端口 |

## 网络准备

确保两台服务器之间以下端口可达：

- **6379** — Redis（主节点的 Redis 供副节点 worker1 连接）
- **29500** — torchrun NCCL rendezvous 端口
- **8008** — API HTTP 端口（对外服务）

> **重要**：worker0 和 worker1 使用 `network_mode: host`，torchrun 直接绑定宿主机端口。
> 这是 PyTorch 分布式训练的标准做法，Docker bridge 网络会导致 NCCL 连接超时。

验证连通性：

```bash
# 在副节点上测试
ping <主节点IP>
nc -zv <主节点IP> 6379   # Redis
nc -zv <主节点IP> 29500  # torchrun rendezvous
```

---

## 主节点部署

### 1. 克隆仓库

```bash
git clone https://github.com/lin285170/Wan2.2.git
cd Wan2.2
```

### 2. 配置环境变量

```bash
cp docker/compose.env.example .env
```

编辑 `.env`：

```bash
# 必填：API 密钥（逗号分隔支持多个）
WAN_SERVE_API_KEYS=sk-your-secret-key

# 模型权重父目录（挂载到容器 /ckpt，每个模型在子目录中）
# 目录结构如下：
#   /data/models/Wan2.2-T2V-A14B/
#   /data/models/Wan2.2-I2V-A14B/
#   /data/models/Wan2.2-TI2V-5B/
#   /data/models/Wan2.2-Animate-14B/
#   /data/models/Wan2.2-S2V-14B/
# 只需要下载你实际使用的模型，其余子目录可以不存在。
# 请求时根据 model 字段自动定位到对应子目录。
# 也可以通过 parameters.ckpt_dir 手动指定其他路径。
WAN_CKPT_HOST_PATH=/data/models

# 双节点拓扑：2节点 × 4GPU = 8 GPU 总计
WAN_NNODES=2
WAN_NPROC_PER_NODE=4

# 主节点真实IP（worker0 使用 host 网络，直接绑定宿主机端口）
# 必须使用真实IP，不能用 0.0.0.0 或 127.0.0.1
WAN_MASTER_ADDR=10.0.0.1
WAN_MASTER_PORT=29500

# worker0 使用 host 网络，通过 localhost 连接 Redis
WAN_REDIS_URL_LOCAL=redis://127.0.0.1:6379/0

# API 对外端口
WAN_API_PORT=8008
```

### 3. 启动主节点服务

```bash
docker compose up -d --build
```

这会启动 3 个容器：

- **redis** — Redis 数据库（端口 6379 对外暴露，供副节点连接）
- **api** — FastAPI HTTP 服务（端口 8008）
- **worker0** — GPU worker（node_rank=0，从 Redis 队列取任务）

### 4. 验证主节点

```bash
# 检查容器状态
docker compose ps

# 检查 API 健康状态
curl http://localhost:8008/healthz

# 查看日志
docker compose logs -f api
docker compose logs -f worker0
```

---

## 副节点部署

### 1. 克隆仓库（同一代码版本）

```bash
git clone https://github.com/lin285170/Wan2.2.git
cd Wan2.2
```

### 2. 配置环境变量

```bash
cp docker/compose.env.example .env
```

编辑 `.env`：

```bash
# 与主节点保持一致
WAN_SERVE_API_KEYS=sk-your-secret-key

# 模型权重父目录（副节点上的路径，与主节点相同的目录结构）
WAN_CKPT_HOST_PATH=/data/models

# 双节点拓扑
WAN_NNODES=2
WAN_NPROC_PER_NODE=4

# 主节点实际IP（不是 0.0.0.0，是真实IP）
WAN_MASTER_ADDR=10.0.0.1
WAN_MASTER_PORT=29500

# Redis 连接主节点（关键！）
WAN_REDIS_URL=redis://10.0.0.1:6379/0
```

### 3. 启动副节点服务

```bash
docker compose -f docker-compose.worker.yml up -d --build
```

这会启动 1 个容器：

- **worker1** — GPU worker（node_rank=1，通过 Redis pub/sub 接收信号）

### 4. 验证副节点

```bash
docker compose -f docker-compose.worker.yml ps
docker compose -f docker-compose.worker.yml logs -f worker1
```

日志应显示：`Worker node started; waiting for signals on wan:signal`

---

## 发起视频生成请求

### 支持的模型

| 模型 | model 值 | 必填 input 字段 | 默认 size | 自动 ckpt_dir |
|------|----------|-----------------|-----------|---------------|
| T2V | `wan2.2-t2v-a14b` | prompt | 1280\*720 | `/ckpt/Wan2.2-T2V-A14B` |
| I2V | `wan2.2-i2v-a14b` | prompt + image | 832\*480 | `/ckpt/Wan2.2-I2V-A14B` |
| TI2V | `wan2.2-ti2v-5b` | prompt（image 可选） | 1280\*704 | `/ckpt/Wan2.2-TI2V-5B` |
| Animate | `wan2.2-animate-14b` | prompt + video | 720\*1280 | `/ckpt/Wan2.2-Animate-14B` |
| S2V | `wan2.2-s2v-14b` | prompt + image + audio（或 enable_tts） | 832\*480 | `/ckpt/Wan2.2-S2V-14B` |

> **模型切换说明**：只需在请求的 `model` 字段指定不同的模型 ID，系统会自动定位对应的权重目录。无需重启服务或修改配置。如需自定义权重路径，可通过 `parameters.ckpt_dir` 覆盖。

### T2V — 文本生成视频

```bash
curl -X POST http://10.0.0.1:8008/api/v1/video/generation \
  -H "Authorization: Bearer sk-your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "wan2.2-t2v-a14b",
    "input": {
      "prompt": "A cat walking on a beach at sunset"
    },
    "parameters": {
      "size": "1280*720",
      "frame_num": 81
    }
  }'
```

### I2V — 图片生成视频

```bash
curl -X POST http://10.0.0.1:8008/api/v1/video/generation \
  -H "Authorization: Bearer sk-your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "wan2.2-i2v-a14b",
    "input": {
      "prompt": "A cat dancing on the beach",
      "image": "/ckpt/ref_image.jpg"
    },
    "parameters": {
      "size": "832*480"
    }
  }'
```

### TI2V — 文本/图片生成视频（5B 轻量模型）

```bash
curl -X POST http://10.0.0.1:8008/api/v1/video/generation \
  -H "Authorization: Bearer sk-your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "wan2.2-ti2v-5b",
    "input": {
      "prompt": "A dog running in a park",
      "image": "/ckpt/ref_image.jpg"
    },
    "parameters": {
      "size": "1280*704"
    }
  }'
```

### Animate — 姿态驱动生成视频

```bash
curl -X POST http://10.0.0.1:8008/api/v1/video/generation \
  -H "Authorization: Bearer sk-your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "wan2.2-animate-14b",
    "input": {
      "prompt": "视频中的人在做动作",
      "video": "/ckpt/ref_video.mp4"
    },
    "parameters": {
      "size": "720*1280",
      "src_root_path": "/ckpt/animate_input",
      "refert_num": 77
    }
  }'
```

### S2V — 语音驱动生成视频

使用音频文件：

```bash
curl -X POST http://10.0.0.1:8008/api/v1/video/generation \
  -H "Authorization: Bearer sk-your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "wan2.2-s2v-14b",
    "input": {
      "prompt": "A person talking happily",
      "image": "/ckpt/ref_image.jpg",
      "audio": "/ckpt/speech.wav"
    },
    "parameters": {
      "size": "832*480"
    }
  }'
```

使用 TTS 合成语音：

```bash
curl -X POST http://10.0.0.1:8008/api/v1/video/generation \
  -H "Authorization: Bearer sk-your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "wan2.2-s2v-14b",
    "input": {
      "prompt": "A person talking happily",
      "image": "/ckpt/ref_image.jpg"
    },
    "parameters": {
      "size": "832*480",
      "enable_tts": true,
      "tts_prompt_audio": "/ckpt/prompt_voice.wav",
      "tts_prompt_text": "希望你以后能够做的比我还好呦。",
      "tts_text": "收到好友从远方寄来的生日礼物，那份意外的惊喜让我心中充满了甜蜜的快乐。"
    }
  }'
```

### 通用：返回格式

所有请求成功后返回：

```json
{
  "request_id": "...",
  "output": {
    "task_id": "wan-..."
  }
}
```

### 查询任务状态

```bash
curl http://10.0.0.1:8008/api/v1/tasks/wan-xxxx \
  -H "Authorization: Bearer sk-your-secret-key"
```

### 下载视频

```bash
curl http://10.0.0.1:8008/api/v1/files/by-task/wan-xxxx \
  -H "Authorization: Bearer sk-your-secret-key" \
  -o output.mp4
```

---

## 工作流程

```
用户 → API(:8008) → Redis 队列
                          ↓
                    worker0 (master) 从队列取任务
                          ↓
                    worker0 写入 job JSON
                    worker0 通过 Redis pub/sub 发信号 → worker1 收到信号
                    worker0 本地执行 torchrun (node_rank=0)    worker1 本地执行 torchrun (node_rank=1)
                          ↓                                         ↓
                    两节点通过 NCCL rendezvous (:29500) 相遇，分布式训练
                          ↓
                    worker0 更新 Redis 任务状态 → SUCCEEDED
                          ↓
用户查询状态 → API 读取 Redis → 返回结果
```

---

## 常用运维命令

```bash
# 停止所有服务（主节点）
docker compose down

# 停止所有服务（副节点）
docker compose -f docker-compose.worker.yml down

# 重启单个服务
docker compose restart api

# 查看资源使用
docker stats

# 清理并重建
docker compose down -v  # 注意：-v 会删除 Redis 数据卷
docker compose up -d --build
```

---

## 单节点模式（测试用）

如果只有一台机器，修改 `.env`：

```bash
WAN_NNODES=1
WAN_NPROC_PER_NODE=4
```

只启动主节点即可（worker1 不需要）：

```bash
docker compose up -d --build
```