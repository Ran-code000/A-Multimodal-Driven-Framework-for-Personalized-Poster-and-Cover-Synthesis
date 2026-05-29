# 图文驱动的个性化海报与封面生成系统

> A Multimodal-Driven Framework for Personalized Poster and Cover Synthesis

基于 **SDXL** 扩散模型，融合 **ControlNet-Canny** 布局控制与 **IP-Adapter** 风格迁移，结合 **DeepSeek** 大语言模型提示词增强，实现文本+布局+风格三模态协同可控的高质量海报生成。

---

## 目录

- [系统架构](#系统架构)
- [环境要求](#环境要求)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [使用说明](#使用说明)
- [项目结构](#项目结构)
- [评测与实验](#评测与实验)
- [低显存优化](#低显存优化)
- [常见问题](#常见问题)
- [参考文献](#参考文献)

---

## 系统架构

```
用户输入（主题/草图/参考图）
        │
        ▼
┌──────────────────┐
│  Gradio Web UI   │  ← 交互界面
└──────────────────┘
        │
        ├──► DeepSeek API ──► LLM 提示词增强
        │
        ├──► Canny 边缘检测 ──► ControlNet 布局条件注入
        │
        └──► CLIP 视觉编码 ──► IP-Adapter 风格条件注入
                    │
                    ▼
        ┌──────────────────┐
        │  SDXL UNet 2.6B  │  ← 扩散去噪核心
        └──────────────────┘
                    │
                    ▼
        ┌──────────────────┐
        │  VAE 解码器      │  ← 生成 1024×1024 海报
        └──────────────────┘
```

**五大核心模块：**

| 模块 | 功能 | 技术方案 |
|------|------|---------|
| LLM 提示词增强 | 将简短主题扩充为高质量 SDXL 提示词 | DeepSeek-Chat API |
| Canny 边缘检测 | 提取布局草图的结构信息 | OpenCV Canny |
| ControlNet 布局注入 | 以弱约束方式引导构图布局 | ControlNet-Canny (SDXL) |
| IP-Adapter 风格注入 | 迁移参考图的视觉风格特征 | IP-Adapter-Plus (SDXL) |
| SDXL 扩散去噪 | 多模态条件协同生成 | SDXL Base 1.0 |

---

## 环境要求

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| GPU | NVIDIA RTX 3060 (12 GB VRAM) | NVIDIA RTX 5060 Ti (16 GB VRAM) |
| 内存 | 16 GB DDR4 | 32 GB DDR5 |
| 存储 | 30 GB 可用空间 | 50 GB NVMe SSD |
| 操作系统 | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |

### 软件要求

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| Python | 3.10 -- 3.11 | 推荐 3.10.13 |
| CUDA | 12.1+ | GPU 加速 |
| PyTorch | 2.1+ | 深度学习框架 |
| Diffusers | 0.25+ | HuggingFace 扩散模型库 |
| Gradio | 4.0+ | Web 交互界面 |

---

## 安装指南

### 1. 克隆仓库

```bash
git clone <your-repo-url>
cd "A-Multimodal-Driven-Framework-for-Personalized-Poster-and-Cover-Synthesis"
```

### 2. 创建虚拟环境（推荐）

```bash
# 使用 conda
conda create -n poster python=3.11 -y
conda activate poster

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置 API 密钥（可选）

提示词增强功能需要 DeepSeek API 密钥。如果不配置，系统仍可运行（手动输入提示词即可）。

```bash
# Windows PowerShell
$env:DEEPSEEK_API_KEY = "sk-your-key-here"

# Linux / macOS
export DEEPSEEK_API_KEY="sk-your-key-here"
```

### 5. 模型下载

首次运行时，程序将自动从 HuggingFace 下载所需模型（约 12 GB）。如果网络受限，请设置镜像：

```bash
# Windows PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"

# Linux / macOS
export HF_ENDPOINT="https://hf-mirror.com"
```

---

## 快速开始

```bash
# 启动 Web 界面
python app.py
```

浏览器访问 **http://localhost:7860** 即可使用。

### 基本工作流

1. **输入主题** — 在文本框中输入海报主题（如"科技风春季校招主视觉海报"）
2. **增强提示词** — 点击"Enhance Prompt"按钮，DeepSeek 自动扩充为高质量 SDXL 提示词
3. **上传参考图（可选）** — 上传布局草图和/或风格参考图
4. **调整参数（可选）** — 调节 ControlNet 权重 (0.15--0.55) 和 IP-Adapter 权重 (0.30--0.40)
5. **生成海报** — 点击"Generate Poster"，等待 15--25 秒即可获得 1024×1024 海报

---

## 使用说明

### Web 界面

```
┌─────────────┐    ┌──────────────────┐
│  左侧输入栏  │    │   右侧结果展示栏   │
│             │    │                  │
│  主题文本框  │    │   生成海报大图    │
│  增强按钮    │    │                  │
│  布局草图上  │    │   参数回显信息    │
│  风格参考上  │    │                  │
│  参数滑块    │    │   历史生成画廊    │
│  生成按钮    │    │                  │
└─────────────┘    └──────────────────┘
```

### 运行测试

```bash
# 验证流水线加载成功且无 OOM
python test_pipeline.py

# 运行完整评测（16 组实验）
python evaluation.py

# 仅计算已有结果的指标
python compute_metrics.py evaluation_v2_output/
```

### 单张生成脚本

```bash
# 纯文本生成（G1 基线）
python gen_one.py --prompt "your prompt here" --output output.png

# 完整多模态生成（G4）
python gen_one.py --prompt "your prompt" --layout sketch.png --style ref.jpg --output output.png
```

---

## 项目结构

```
.
├── engine.py                    # 核心生成引擎（PosterEngine 类）
├── llm_client.py                # DeepSeek 提示词增强客户端
├── app.py                       # Gradio Web 交互界面
├── gen_one.py                   # 单张海报生成脚本
├── gen_v2.py                    # 批量生成脚本
├── evaluation.py                # 自动化评测流水线
├── compute_metrics.py           # CLIP Score / FID / 美学指标计算
├── test_pipeline.py             # 流水线验证与 OOM 检测
├── requirements.txt             # Python 依赖清单
├── README.md                    # 本文件
│
├── evaluation_v2_output/        # 16 组评测生成结果
│   ├── ancient-beauty_G1--G4/   # 古风美人 ×4 组
│   ├── green-landscape_G1--G4/  # 青绿山水 ×4 组
│   ├── cyberpunk_G1--G4/        # 赛博朋克 ×4 组
│   ├── spring-flowers_G1--G4/   # 春日花卉 ×4 组
│   ├── grid_*.png               # 2×2 主题对比网格
│   ├── summary_grid.png         # 4×4 主汇总对比图
│   └── metrics.json             # 全量定量指标
│
├── test_output_text_only.png    # 纯文本基线生成样例
├── test_output_full_pipeline.png# 完整多模态生成样例
├── ui.png                       # Gradio 界面截图
│
├── proposal.pdf                 # 开题报告（PDF）
├── final_report.pdf             # 期末结题报告（PDF）
└── demo_video_link.txt          # Demo 演示视频链接
```

---

## 评测与实验

### 对照实验设计

| 实验组 | ControlNet | IP-Adapter | CN 权重 | IP 权重 |
|--------|:----------:|:----------:|:-------:|:-------:|
| G1 纯文本基线 | ✗ | ✗ | 0.0 | 0.0 |
| G2 仅布局控制 | ✓ | ✗ | 0.55 | 0.0 |
| G3 仅风格控制 | ✗ | ✓ | 0.0 | 0.40 |
| G4 完整多模态 | ✓ | ✓ | 0.55 | 0.40 |

### 评测主题

| 主题 | 风格特征 | 构图类型 |
|------|---------|---------|
| 古风美人 | 暖色调古典人物画 | 中央人物+四角框架 |
| 青绿山水 | 蓝绿色山水画 | 层叠山峦+水面+亭台 |
| 赛博朋克 | 紫青色霓虹城市 | 透视网格+竖向建筑 |
| 春日花卉 | 粉色调花卉摄影 | 圆形花簇散点+地面基线 |

### 核心结果

| 指标 | G1 基线 | G4 完整系统 | 提升 |
|------|--------|-----------|------|
| CLIP Score | 0.304 | **0.341** | +12.2% |
| Aesthetic Score | 2.83 | **3.53** | +24.7% |
| 主观美观度 | 2.7 | **4.1** | +51.9% |

---

## 低显存优化

本系统面向 16 GB VRAM 消费级 GPU 进行了深度优化，组合策略如下：

| 优化技术 | 显存节约 | 说明 |
|---------|---------|------|
| FP16 半精度推理 | ~50% | 所有模型以 float16 加载 |
| CPU 卸载 | ~40%（总峰值） | 非活跃组件常驻 CPU，按需迁移 GPU |
| VAE 切片/瓦片 | ~60%（VAE 峰值） | 分块编解码高分辨率图像 |
| torch.inference_mode | 消除中间激活 | 彻底禁用 autograd 引擎 |
| xFormers 高效注意力 | ~15--20% | 优化自注意力计算 |

优化后静态显存约 6--8 GB，峰值约 11--13 GB，为推理预留充足安全余量。

---

## 常见问题

**Q: 启动时显存不足（OOM）？**
A: 确保已启用 CPU 卸载和 VAE 优化（`engine.py` 中默认开启）。关闭其他 GPU 占用程序后再试。

**Q: 模型下载失败？**
A: 设置 HuggingFace 镜像：`HF_ENDPOINT=https://hf-mirror.com`。如仍失败，可手动下载模型至 `~/.cache/huggingface/hub/`。

**Q: 生成速度慢？**
A: 将推理步数从 25 降至 15（调试模式），或探索 LCM-LoRA 加速（未来工作）。

**Q: 不需要提示词增强？**
A: 直接在增强提示词框中输入 SDXL 提示词即可，跳过 DeepSeek API 调用。

---

## 参考文献

1. Podell D, et al. *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis.* arXiv, 2023.
2. Zhang L, et al. *Adding Conditional Control to Text-to-Image Diffusion Models.* ICCV, 2023.
3. Ye H, et al. *IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models.* arXiv, 2023.
4. Rombach R, et al. *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR, 2022.

---

## 作者

- **姓名**: 崔敬然
- **学号**: 23336055
- **课程**: 多模态大模型原理与应用

---

## 许可证

本项目仅用于学术研究与课程作业。所使用预训练模型（SDXL、ControlNet、IP-Adapter）遵循其各自的许可证条款。
