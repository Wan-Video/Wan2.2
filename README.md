# Wan2.2 集群推理与 DashScope 风格 HTTP 服务部署指南

本文说明如何在 **2 台 × 4×A100 40GB**（或任意 `nnodes × nproc_per_node = WORLD_SIZE`）上运行本仓库自带的 **异步任务 API**（兼容 DashScope 习惯的 `Bearer` 鉴权、`POST` 提交、`GET` 轮询任务状态）。

---

## 1. 架构说明

| 组件 | 职责 |
|------|------|
| **Redis** | 任务队列 `WAN_QUEUE_NAME`、任务元数据 `WAN_TASK_KEY_PREFIX*`、集群互斥锁 `WAN_CLUSTER_LOCK_KEY`（同一时刻只跑一个 `torchrun` 作业）。 |
| **`run_api_server.py` + `serve.api`** | FastAPI：提交任务、查询状态、下载 MP4。 |
| **`python -m serve.worker_main`** | 从队列取 `task_id`，写 `job.json`，调用 `torchrun … generate_job.py`。 |
| **`generate_job.py`** | 读取 JSON，调用 `generate.args_from_job_dict` + `generate.generate`。 |

**重要**：默认实现假设 **只有一个 worker 进程** 在消费队列（全局 GPU 锁）。若启动多个 worker 会抢锁并反复 requeue；多任务并发需改造为每套 GPU 独立队列或 Kubernetes Job。

---

## 2. 环境准备（两台 GPU 机 + 一台 API 机可选）

1. **Python**：与官方 README 一致，`torch>=2.4`，安装 `requirements.txt` + 推理所需依赖。  
2. **服务依赖**（跑 API / worker 的机器）：  
   ```bash
   pip install -r requirements_serve.txt
   ```  
3. **Redis**：可部署在 API 同机或独立 VM；两台 GPU 机与 API 均需能访问该地址。  
4. **共享存储（强烈推荐）**：NFS 等，两台 GPU 上 **相同绝对路径** 挂载：  
   - 模型目录 `WAN_CKPT_DIR`  
   - 任务 JSON 目录 `WAN_JOB_DIR`  
   - 输出视频目录 `WAN_OUTPUT_DIR`  

5. **NCCL 双机**：设置 `NCCL_SOCKET_IFNAME`、主机名解析、防火墙放行 `WAN_MASTER_PORT` 及 PyTorch 分布式端口；有 RDMA 时按机房文档配置 IB。

---

## 3. 环境变量参考

### 通用 / API / Worker

| 变量 | 说明 | 示例 |
|------|------|------|
| `WAN_SERVE_API_KEYS` | 逗号分隔的 API Key（`Authorization: Bearer <key>`） | `sk-local-xxx,sk-local-yyy` |
| `WAN_REDIS_URL` | Redis 连接串 | `redis://10.0.0.5:6379/0` |
| `WAN_REPO_ROOT` | 本仓库绝对路径 | `/data/Wan2.2` |
| `WAN_JOB_DIR` | 任务 JSON 目录（需共享） | `/mnt/wan/jobs` |
| `WAN_OUTPUT_DIR` | 输出 MP4（需共享） | `/mnt/wan/out` |
| `WAN_CKPT_DIR` | 默认 checkpoint 根目录 | `/mnt/wan/Wan2.2-T2V-A14B` |

### 多机 torchrun（Worker 所在机应能 `ssh` 到第二台时）

| 变量 | 说明 |
|------|------|
| `WAN_NNODES` | 节点数，例如 `2` |
| `WAN_NPROC_PER_NODE` | 每节点进程数，例如 `4`（总 8 卡） |
| `WAN_MASTER_ADDR` | rank0 所在机 IP（**第一**台 GPU 机） |
| `WAN_MASTER_PORT` | rendezvous 端口，如 `29500` |
| `WAN_RDZV_PREFIX` | rendezvous id 前缀（会再拼 `task_id`） |
| `WAN_SSH_SECOND_NODE` | 第二台登录串，如 `ubuntu@192.168.1.12` |
| `WAN_SSH_TORCHRUN_PREFIX` | SSH 远端 shell 前缀，默认 `cd {repo_root} && export PYTHONPATH={repo_root}:$PYTHONPATH && ` |
| `WAN_PYTHON` / `WAN_TORCHRUN` | 可选，覆盖可执行文件路径 |

### Prompt 扩展（可选）

若任务 JSON 里 `use_prompt_extend=true` 且 `prompt_extend_method=dashscope`：

- `DASH_API_KEY`  
- 国际站可设 `DASH_API_URL=https://dashscope-intl.aliyuncs.com/api/v1`

---

## 4. 单机 8 卡（单节点测试）

```bash
export WAN_SERVE_API_KEYS="sk-dev"
export WAN_REDIS_URL="redis://127.0.0.1:6379/0"
export WAN_REPO_ROOT="/data/Wan2.2"
export WAN_CKPT_DIR="/data/Wan2.2-T2V-A14B"
export WAN_JOB_DIR="/tmp/wan_jobs"
export WAN_OUTPUT_DIR="/tmp/wan_out"
export WAN_NNODES=1
export WAN_NPROC_PER_NODE=8
export PYTHONPATH="/data/Wan2.2:$PYTHONPATH"

# 终端 1
redis-server &
python run_api_server.py

# 终端 2（与 API 同机或能访问 Redis 的 GPU 机）
python -m serve.worker_main
```

提交示例：

```bash
curl -sS -X POST "http://127.0.0.1:8008/api/v1/video/generation" \
  -H "Authorization: Bearer sk-dev" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "wan2.2-t2v-a14b",
    "input": { "prompt": "A cat walking on grass." },
    "parameters": {
      "size": "1280*720",
      "dit_fsdp": true,
      "t5_fsdp": true,
      "ulysses_size": 8,
      "offload_model": false,
      "convert_model_dtype": true
    }
  }'
```

查询与下载（将 `TASK_ID` 换成响应里的 `task_id`）：

```bash
curl -sS -H "Authorization: Bearer sk-dev" \
  "http://127.0.0.1:8008/api/v1/tasks/TASK_ID"

curl -L -o out.mp4 -H "Authorization: Bearer sk-dev" \
  "http://127.0.0.1:8008/api/v1/files/by-task/TASK_ID"
```

---

## 5. 双机 2×4 卡 A100（推荐生产形态）

下文假设 **GPU 节点 0**（主节点，跑 Worker + 本地 `torchrun`）与 **GPU 节点 1**（从节点，仅通过 SSH 被拉起 `torchrun`）各 **4×A100 40GB**，合计 **8 卡** 跑 `t2v-A14B` / `i2v-A14B` 等需 `WORLD_SIZE=8` 且 `ulysses_size=8` 的任务。`serve/launcher.py` 在 `WAN_NNODES>1` 时会在 **节点 0 本机** 启动 `torchrun`，并通过 **SSH** 在 **节点 1** 启动 **完全相同** 的一条 `torchrun` 命令，由 PyTorch **c10d rendezvous** 完成组网。

### 5.1 拓扑与角色

| 角色 | 建议部署位置 | 说明 |
|------|----------------|------|
| **Redis** | 第三台小规格机器、或节点 0、或托管云服务 | API 与 Worker 均需 `WAN_REDIS_URL` 可达。 |
| **HTTP API** | 任意能访问 Redis 的机器（可无 GPU） | `run_api_server.py`，对客户端暴露 `8008`。 |
| **GPU Worker** | **仅节点 0 上跑一个进程** | `python -m serve.worker_main`；默认全局 GPU 锁，不要双机各起一个 Worker 消费同一队列。 |
| **推理进程** | 节点 0：本地 `torchrun`；节点 1：经 SSH 启动的 `torchrun` | 两机 `torchrun` 参数一致，`--rdzv_endpoint` 指向 **节点 0 可达 IP**。 |

### 5.2 网络与主机名

1. 为两机分配固定内网 IP，例如：节点 0 → `10.0.0.10`，节点 1 → `10.0.0.11`。  
2. `WAN_MASTER_ADDR` 必须填 **节点 0 上对节点 1 可达的 IP**（通常即 `10.0.0.10`），**不要**填 `127.0.0.1`。  
3. 开放防火墙：**`WAN_MASTER_PORT`（如 29500）** 以及 PyTorch/NCCL 可能使用的端口段（或先临时放宽双机间 TCP 以便联调）。  
4. 若跨机 RDMA，按机房规范配置 IB；仅用 TCP 时可先设 `export NCCL_IB_DISABLE=1` 排除 IB 干扰（性能会下降，仅用于排障）。

### 5.3 共享存储（NFS 或并行文件系统）

两机对以下路径使用 **同一挂载点、同一绝对路径**（示例均为 `/mnt/wan/...`，可按机房替换）：

| 路径 | 用途 |
|------|------|
| `WAN_REPO_ROOT`（如 `/mnt/wan/Wan2.2`） | 本仓库代码，两机一致。 |
| `WAN_CKPT_DIR`（如 `/mnt/wan/Wan2.2-T2V-A14B`） | 模型权重只读；Worker 内常为 `/ckpt`，宿主机挂载需与 `WAN_CKPT_DIR` 一致。 |
| `WAN_JOB_DIR` | 任务 JSON；Worker 写入，`job_json` 为 NFS 路径以便两机 `torchrun` 同读。 |
| `WAN_OUTPUT_DIR` | 生成 MP4；仅 rank 0 写盘，放 NFS 便于 API 机或节点 0 取文件。 |

挂载后分别在两机执行：`ls -la $WAN_REPO_ROOT/generate_job.py` 与 `ls $WAN_CKPT_DIR`，确认路径一致、权限可读。

### 5.4 软件环境（两机必须对齐）

1. **操作系统与驱动**：两机安装同一主线版本 **NVIDIA 驱动**，`nvidia-smi` 正常。  
2. **Python**：建议 **同版本**（如 3.10/3.11），各自 `venv` 或 **同一套 Conda env** 的克隆亦可，关键是 **`torch` 版本与 CUDA 构建一致**。  
3. **依赖**：两机均在 `WAN_REPO_ROOT` 下执行 `pip install -r requirements.txt` 与 `pip install -r requirements_serve.txt`（`flash_attn` 若装不上可先跳过，与单机排障相同）。  
4. **`torchrun` 在 PATH 中**：`which torchrun` 两机均有结果。  

节点 1 **不跑** `serve.worker_main`，但必须能通过 SSH 执行与节点 0 **相同**的 `torchrun … generate_job.py`，因此节点 1 也需完整 Python 环境与仓库代码（与节点 0 同一路径最省事）。

### 5.5 节点 0 → 节点 1 免密 SSH

在 **节点 0** 上（以运行 Worker 的 Linux 用户执行）：

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519_wan -C "wan-worker"
# 将公钥追加到节点 1 的 authorized_keys（把 user、10.0.0.11 换成实际值）
ssh-copy-id -i ~/.ssh/id_ed25519_wan.pub user@10.0.0.11
# 若使用自定义 key：
ssh -i ~/.ssh/id_ed25519_wan user@10.0.0.11 'hostname'
```

`WAN_SSH_SECOND_NODE` 建议写成 **`user@10.0.0.11`**，与上述 `ssh` 登录串一致。  
生产环境建议在 `serve/launcher.py` 中为 `ssh` 增加 `KnownHostsFile` / 关闭 `StrictHostKeyChecking=no`，避免中间人风险。

### 5.6 NCCL 与常见环境变量（两机 Worker 进程继承；SSH 子进程同理）

在 **节点 0** 启动 Worker 的 shell 或 systemd 中可导出（按网卡名修改）：

```bash
export NCCL_SOCKET_IFNAME=eth0      # 或 ens、bond0 等，双机互通的网卡
export NCCL_DEBUG=WARN              # 排障时可改为 INFO
# export NCCL_IB_DISABLE=1          # 无 IB 或联调时可开
```

### 5.7 双机专用环境变量（节点 0 上配置）

以下变量在 **跑 `python -m serve.worker_main` 的节点 0** 上设置（可写入 `/etc/default/wan-worker` 或 systemd `Environment=`）：

```bash
export WAN_NNODES=2
export WAN_NPROC_PER_NODE=4
export WAN_MASTER_ADDR=10.0.0.10       # 节点 0 对外的内网 IP
export WAN_MASTER_PORT=29500
export WAN_SSH_SECOND_NODE=user@10.0.0.11

export WAN_REPO_ROOT=/mnt/wan/Wan2.2
export PYTHONPATH=/mnt/wan/Wan2.2:$PYTHONPATH
export WAN_CKPT_DIR=/mnt/wan/Wan2.2-T2V-A14B
export WAN_JOB_DIR=/mnt/wan/jobs
export WAN_OUTPUT_DIR=/mnt/wan/out

export WAN_REDIS_URL=redis://10.0.0.5:6379/0
export WAN_SERVE_API_KEYS=sk-your-secret

# 可选：保持默认即可；{repo_root} 会替换为 WAN_REPO_ROOT
# export WAN_SSH_TORCHRUN_PREFIX='cd {repo_root} && export PYTHONPATH={repo_root}:$PYTHONPATH && '
```

说明：

- **`WAN_NNODES` × `WAN_NPROC_PER_NODE` = 8** 时，任务 JSON / API 里 **`ulysses_size` 必须为 8**，且 **`dit_fsdp` / `t5_fsdp`** 与官方多卡示例一致。  
- **`WAN_MASTER_PORT`** 在每次作业中由 `rdzv_id`（含 `task_id`）区分不同 rendezvous；端口需空闲。  
- **`WAN_SSH_TORCHRUN_PREFIX`** 中的 **`{repo_root}`** 由程序替换为 `WAN_REPO_ROOT` 的绝对路径（见 `serve/config.py`）。

### 5.8 启动顺序（推荐）

1. **启动 Redis**（若尚未运行）。  
2. **启动 API**（可在无 GPU 的机器上）：  
   `export WAN_REDIS_URL=...` 等与队列、路径相关变量后执行 `python run_api_server.py`。  
3. **仅在节点 0 启动 Worker**：  
   ```bash
   cd "$WAN_REPO_ROOT"
   export PYTHONPATH="$WAN_REPO_ROOT:$PYTHONPATH"
   python -m serve.worker_main
   ```  
4. 用 **curl** 提交一条任务（见上文 §4），观察 Worker 日志：应先出现本地 `torchrun`，约 2 秒后出现 SSH 在节点 1 起的第二条 `torchrun`，最后 rank 0 写 `save_file`。

### 5.9 行为说明（与源码一致）

`serve/launcher.py` 在 `WAN_NNODES>1` 时：

1. 用 `subprocess.Popen` 在 **节点 1** 上执行：  
   `ssh … user@node1 'bash -lc "<WAN_SSH_TORCHRUN_PREFIX><torchrun 完整命令>"'`  
2. **约 2 秒** 后在 **节点 0** 上 `subprocess.run` 同样的 `torchrun` 命令。  
3. 两条命令中的 **`--job_json` 为 NFS 上的同一文件**；**`--rdzv_id` 每次作业唯一**（含 `task_id`），避免与历史进程冲突。

### 5.10 排障清单

| 现象 | 检查项 |
|------|--------|
| SSH 失败 | 节点 0 上手动 `ssh user@node1`；`ssh-agent`、私钥权限、`authorized_keys`。 |
| rendezvous 超时 / 挂住 | `WAN_MASTER_ADDR` 是否可从节点 1 `telnet`/`nc -zv` 到端口；防火墙；两机时钟是否大致同步（建议 NTP）。 |
| NCCL 报错 | `NCCL_SOCKET_IFNAME`；必要时 `NCCL_IB_DISABLE=1` 试跑。 |
| 节点 1 找不到模块 | 节点 1 上 `PYTHONPATH` 与 `cd` 是否与 `WAN_SSH_TORCHRUN_PREFIX` 一致；`pip show torch`。 |
| 仅单机起进程 | `WAN_SSH_SECOND_NODE` 是否为空；`WAN_NNODES` 是否仍为 1。 |
| OOM / 显存 | 40GB×4 跑 A14B 需 FSDP+Ulysses 与合适 `offload_model` / `convert_model_dtype`，与官方 README 多卡说明一致。 |

### 5.11 与 Docker 的关系

`docker-compose.yml` 默认描述 **单机多卡容器**。双机物理机 + SSH `torchrun` 时，通常做法是：**不在节点 1 上再跑一个消费同一 Redis 队列的 Worker 容器**；仅在 **节点 0** 起 Worker（裸机或单容器），并配置 `WAN_SSH_SECOND_NODE` 指向节点 1 的 **SSH 可达地址**，且两机挂载 **同一 NFS** 到相同路径。若两机都跑在容器内，还需保证 **容器到容器/宿主 SSH**、以及 **容器内 `WAN_MASTER_ADDR` 对另一机可见**（常用 host 网络或显式端口映射，视编排而定）。

---

## 6. 任务 JSON 与 `model` 别名

HTTP 请求体会被合并为 `generate.args_from_job_dict` 可接受的字典：

- 顶层 **`model`** 可为：`wan2.2-t2v-a14b`、`wan2.2-i2v-a14b`、`wan2.2-ti2v-5b`、`wan2.2-s2v-14b`、`wan2.2-animate-14b`，或直接 `WAN_CONFIGS` 里的 `task` 字符串。  
- 其余字段与 `generate.py` 命令行一致，嵌套在 `input` / `parameters` 中亦可。  
- `sample_guide_scale` 可为 **单个 float** 或 **两个 float 的数组**（低/高噪声专家）。  

直接调用 `generate_job.py`（不经 HTTP）示例：

```bash
cat > /mnt/wan/jobs/manual.json <<'EOF'
{
  "model": "wan2.2-t2v-a14b",
  "ckpt_dir": "/mnt/wan/Wan2.2-T2V-A14B",
  "save_file": "/mnt/wan/out/manual.mp4",
  "prompt": "Two cats boxing on stage.",
  "size": "1280*720",
  "dit_fsdp": true,
  "t5_fsdp": true,
  "ulysses_size": 8,
  "convert_model_dtype": true,
  "offload_model": false
}
EOF

torchrun --nnodes=1 --nproc_per_node=8 --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:29501 --rdzv_id=manual1 \
  /mnt/wan/Wan2.2/generate_job.py --job_json /mnt/wan/jobs/manual.json
```

---

## 7. systemd 示例（API）

`/etc/systemd/system/wan-api.service`：

```ini
[Unit]
Description=Wan2.2 HTTP API
After=network.target

[Service]
User=wan
WorkingDirectory=/mnt/wan/Wan2.2
Environment=PYTHONPATH=/mnt/wan/Wan2.2
Environment=WAN_SERVE_API_KEYS=sk-prod-xxx
Environment=WAN_REDIS_URL=redis://127.0.0.1:6379/0
Environment=WAN_CKPT_DIR=/mnt/wan/Wan2.2-T2V-A14B
Environment=WAN_JOB_DIR=/mnt/wan/jobs
Environment=WAN_OUTPUT_DIR=/mnt/wan/out
Environment=WAN_REPO_ROOT=/mnt/wan/Wan2.2
ExecStart=/mnt/wan/venv/bin/python /mnt/wan/Wan2.2/run_api_server.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Worker 类似，将 `ExecStart` 改为 `python -m serve.worker_main`，并在 GPU 节点 0 上运行。

---

## 8. 安全与运维建议

- 仅内网暴露 API，或前置 mTLS / 零信任网关。  
- 定期轮换 `WAN_SERVE_API_KEYS`。  
- 大模型与生成结果路径做磁盘配额与清理任务。  
- 监控 Redis 队列长度、worker 日志、`torchrun` 退出码。  

---

## 9. 容器化部署（Docker Compose）

仓库提供 **CPU 版 API 镜像** 与 **GPU Worker 镜像**，由 `docker-compose.yml` 编排 Redis、API、Worker。

### 9.1 前置条件

- 已安装 [Docker](https://docs.docker.com/engine/install/) 与 [Docker Compose V2](https://docs.docker.com/compose/)。  
- **Worker 所在宿主机** 安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)，并可用 `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` 验证。  
- 将官方权重下载到宿主机目录，例如 `/data/Wan2.2-T2V-A14B`，供 **只读** 挂载到 Worker 容器的 `/ckpt`。

### 9.2 配置与启动

```bash
cd /path/to/Wan2.2
cp docker/compose.env.example .env
# 编辑 .env：至少设置 WAN_SERVE_API_KEYS、WAN_CKPT_HOST_PATH
```

仅启动 **Redis + API**（开发机无 GPU时）：

```bash
docker compose up -d --build redis api
```

在 **带 NVIDIA GPU 的机器** 上启动完整栈（含 Worker，使用 Compose `gpu` profile）：

```bash
docker compose --profile gpu up -d --build
```

常用命令：

```bash
docker compose logs -f api worker
docker compose ps
```

API 默认映射到宿主机 `WAN_API_PORT`（默认 `8008`）。健康检查：`GET http://<host>:8008/healthz`。

### 9.3 数据卷说明

| 卷名 | 挂载点 | 说明 |
|------|--------|------|
| `wan_shared` | 容器内 `/data` | `jobs` → `/data/jobs`，`outputs` → `/data/outputs`；API 与 Worker 共享，用于任务 JSON 与生成视频。 |
| 绑定挂载 | `/ckpt` | 来自 `.env` 的 `WAN_CKPT_HOST_PATH`，只读挂载到 Worker。 |

### 9.4 镜像构建参数（Worker）

| 构建参数 | 默认 | 说明 |
|----------|------|------|
| `BASE_IMAGE` | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` | 可按机房 CUDA 版本替换为官方 PyTorch 标签。 |
| `INSTALL_FLASH_ATTN` | `0` | 设为 `1` 时尝试安装 `flash_attn`（需与基础镜像 CUDA 匹配，失败时构建仍可能继续）。 |

示例：

```bash
docker build -f docker/Dockerfile.worker \
  --build-arg INSTALL_FLASH_ATTN=1 \
  -t wan2-worker:latest .
```

### 9.5 双机 GPU 与 Compose

`docker-compose.yml` 描述的是 **单机上的多卡容器**。若要在 **两台物理机** 各跑 4 卡并沿用现有 `serve.launcher` 的 SSH 双机 `torchrun`：

1. 两台机器安装 Docker + NVIDIA Toolkit，**同一 NFS** 挂载到相同路径（含代码、权重、`WAN_JOB_DIR` / `WAN_OUTPUT_DIR`）。  
2. 在 **节点 0** 上可仍用 Compose 起 Redis（或外置托管 Redis），API 与 Worker 容器；在 **节点 1** 仅起 **Worker 容器**（或不用 Compose，直接 `docker run`），两台 Worker 不要同时消费同一队列——当前设计为 **单 worker 消费**；双机多卡推荐 **只在节点 0 起一个 Worker 容器**，并在 `.env` 中配置 `WAN_NNODES=2`、`WAN_NPROC_PER_NODE=4`、`WAN_MASTER_ADDR`、`WAN_SSH_SECOND_NODE`，由容器内 `torchrun` + SSH 拉起第二台进程（需节点 0 容器能 SSH 到节点 1，且节点 1 已安装相同镜像或具备相同 Python/torch 环境）。  

更稳妥的生产方式是将 **Redis + API** 托管在控制面，**每台 GPU 机** 用 `docker run` 或 Kubernetes Job 只跑 `wan2-worker`，并改造队列分区；超出本文范围时可单独扩展。

### 9.6 相关文件

| 路径 | 说明 |
|------|------|
| `docker-compose.yml` | Redis、api、worker 服务定义 |
| `docker/Dockerfile.api` | 仅 FastAPI 依赖的轻量 API 镜像 |
| `docker/Dockerfile.worker` | CUDA + Wan 推理 + `serve.worker` |
| `docker/entrypoint-worker.sh` | Worker 入口 |
| `docker/compose.env.example` | 复制为仓库根目录 `.env` 的模板 |
| `.dockerignore` | 减小构建上下文 |
| `README.md` | 部署与 HTTP 服务主文档（本文件） |
| `DEPLOY_SERVE.md` | 历史/外链兼容：仅指向 `README.md` |

---

## 10. 代码变更摘要

| 路径 | 说明 |
|------|------|
| `README.md` | 部署、DashScope 风格 API、Docker 主文档 |
| `generate.py` | `_build_parser` / `parse_args` / `args_from_job_dict` / `JOB_MODEL_ALIASES` |
| `generate_job.py` | `torchrun` 入口，读 `--job_json` |
| `serve/` | FastAPI、Redis、launcher、worker |
| `run_api_server.py` | 开发用 uvicorn 启动 |
| `requirements_serve.txt` | API 额外依赖 |
| `docker-compose.yml` / `docker/*` | 容器化编排与镜像 |

若需 **HTTPS、限流、多队列、回调 Webhook**，可在 `serve/api.py` 外再包一层网关或扩展本模块。
