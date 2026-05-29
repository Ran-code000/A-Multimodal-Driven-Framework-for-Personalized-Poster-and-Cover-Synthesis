# 图文驱动的个性化海报与封面生成系统
## Image-Text Driven Personalized Poster & Cover Synthesis
### 多模态大模型原理与应用 · 课程大作业开题报告

---

## 1 任务定义与应用背景

### 1.1 任务定义

本课题旨在构建一个**图文驱动的个性化海报与封面生成系统**，核心任务为：给定用户输入的**文本描述**（海报主题）、**布局草图**（构图参考）与**风格参考图**（画风/色彩参考），利用多模态大模型技术自动生成高质量、可控的海报与封面图像。

系统接收三类模态输入：

（1）**文本模态**：用户以自然语言描述海报主题、内容要素与氛围。系统通过大语言模型（DeepSeek）对原始输入进行提示词增强，将其转化为富含艺术术语与构图描述的 SDXL 级高质量提示词。

（2）**布局模态**：用户可手绘或上传简易草图，系统通过 ControlNet-Canny 边缘检测提取结构信息，以弱约束方式引导生成图像的构图布局。

（3）**风格模态**：用户上传风格参考图像，系统通过 IP-Adapter 提取其视觉风格特征并注入生成过程，实现对色彩调性、纹理肌理的可控迁移。

系统输入输出范式：文本提示词 + 布局草图 + 风格参考图 → 多模态生成系统（SDXL+CN+IP-Adapter）→ 个性化海报/封面。

### 1.2 应用背景与挑战

海报与封面设计是视觉传达的核心场景。传统设计流程依赖专业设计师手工制作，周期长、成本高且难以规模化。随着 AIGC 技术的快速发展，基于扩散模型的文生图技术为自动化海报生成提供了新范式。然而，现有方案面临以下关键挑战：

（1）**构图不可控**：纯文本驱动的 SDXL 生成结果构图随机性强，无法满足海报设计对版面结构的精确需求。

（2）**风格不可控**：色彩调性、纹理风格难以通过文本精确描述，生成结果与预期画风偏差大。

（3）**细节崩坏**：人物面部、文字区域等高频细节易出现扭曲变形，商业可用性不足。

（4）**多模态对齐困难**：文本、布局、风格三种异构模态如何在潜在空间中有效融合，是多模态生成的核心难题。

本课题通过 SDXL+ControlNet+IP-Adapter 的协同架构，首次将布局控制与风格迁移统一于个性化海报生成任务，为上述挑战提供系统化解决方案。

## 2 相关工作调研

本节调研五类代表性工作，总结贡献与局限性，明确本文创新定位。

### 2.1 Stable Diffusion XL (SDXL)

SDXL（Podell et al., 2023）是目前最先进的开源文生图基础模型之一。相比 SD 1.5/2.1，SDXL 将参数量提升至 2.6B UNet + 双文本编码器（CLIP ViT-L 与 OpenCLIP ViT-bigG），支持 1024×1024 原生分辨率生成，显著提升了图像质量与文本-图像对齐能力。其开放生态使其成为构建个性化生成系统的理想基座。然而，SDXL 作为通用文生图模型，缺乏对布局和风格的显式控制能力。

### 2.2 ControlNet

ControlNet（Zhang et al., ICCV 2023）提出了一种可插拔的神经网络架构，通过复制 Stable Diffusion UNet 编码器并添加"零卷积"层，在保持原模型生成能力的同时，引入额外条件信号（如 Canny 边缘、OpenPose 骨骼、深度图等）实现空间可控生成。其模块化设计使其可与 SDXL 无缝集成，为本文的布局控制模块提供了技术基础。但其单一条件机制无法同时处理风格信息。

### 2.3 IP-Adapter

IP-Adapter（Ye et al., 2023）提出了一种轻量级图像提示适配器，通过解耦的交叉注意力机制将图像特征注入预训练扩散模型，实现风格/内容参考图的特征迁移。相比 ControlNet 的像素级条件注入，IP-Adapter 在语义/风格层面进行控制，且仅增加约 3% 的参数量。本文采用 IP-Adapter-SDXL 版本作为风格控制的核心模块。

### 2.4 HCP-Diffusion 与个性化文生图

HCP-Diffusion 是一个面向 Stable Diffusion 的微调框架，支持 DreamBooth、LoRA、Textual Inversion 等多种个性化训练方法。在个性化海报生成领域，DreamBooth 与 LoRA 被广泛用于特定风格/人物的定制化生成。然而，基于微调的方案需要为每种风格/主题单独训练，不适用于"零样本"的通用海报生成场景。本文采用无需训练的 IP-Adapter 实现零样本风格迁移。

### 2.5 文本到海报生成相关工作

近期研究者已开始探索 AI 驱动的海报生成。Text2Poster（Yang et al., ICASSP 2022）提出了基于布局预测的海报生成流水线；PosterLayout（Li et al., CVPR 2023）关注海报元素的自动排版优化；GAN-based 方法如 StyleGAN-Poster 在特定领域（如电影海报）取得了一定效果。然而，上述工作普遍面临以下局限：（1）大部分基于 GAN 架构，生成质量低于扩散模型基线；（2）布局和风格控制通常独立进行，缺乏统一的多模态可控框架；（3）模型体量大、推理速度慢，难以在消费级 GPU 上部署。

### 2.6 本文创新点

（1）**首次将 ControlNet 布局控制与 IP-Adapter 风格迁移统一整合至 SDXL 海报生成框架**，实现文本+布局+风格三模态协同可控生成。

（2）**提出弱约束控制策略**：通过降低 ControlNet 权重至 0.15-0.20 范围，在保持构图引导的同时保留 SDXL 的艺术创造力。

（3）**全流程消费级 GPU 适配**：通过模型 CPU 卸载、VAE 切片/瓦片、float16 精度等优化策略，在 16GB VRAM 的消费级 GPU 上实现稳定推理。

## 3 系统方案设计

### 3.1 系统架构

本系统采用模块化流水线架构，包含五个核心模块：提示词增强、布局控制、风格控制、扩散生成与交互界面。

（1）**用户输入层**：接收文本主题、布局草图与风格参考图三种模态输入。

（2）**LLM 提示词增强**：通过 DeepSeek API 将用户简短描述扩充为包含艺术风格、光影描述、构图术语的 SDXL 级高质量提示词。

（3）**布局提取**：对上传的布局草图进行 Canny 边缘检测，提取结构信息作为 ControlNet 条件输入。

（4）**多模态生成核心**：SDXL UNet 同时接收文本嵌入（CLIP+OpenCLIP）、ControlNet 布局条件、IP-Adapter 风格条件，进行扩散去噪。

（5）**VAE 解码与输出**：将潜在表示解码为 1024×1024 最终海报图像，通过 Gradio WebUI 展示。

### 3.2 模型选型

| 组件 | 模型/方案 | 关键参数 |
|------|----------|---------|
| 基座模型 | SDXL Base 1.0 | 2.6B UNet, 1024×1024 |
| 布局控制 | ControlNet-Canny (SDXL) | 权重 0.15-0.20 (弱约束) |
| 风格控制 | IP-Adapter (SDXL) | 权重 0.30-0.40 |
| 提示词增强 | DeepSeek-Chat API | temperature=0.7 |
| 图像编码 | CLIP ViT-L + OpenCLIP | 双编码器串联 |
| 调度器 | DPM-Solver++ (Karras) | 25-30 推理步数 |
| 内存优化 | CPU 卸载 + VAE 切片/瓦片 | fp16, 16GB VRAM |
| 交互界面 | Gradio 6.x | Web UI, 实时预览 |

### 3.3 评测指标设计

本课题采用主观评测 + 客观评测结合的方式对生成质量进行全面评价。

**主观评测指标**：
- 美观度（1-5 分）：1=严重失真/构图混乱；3=基本可接受；5=高度美观、色彩协调、具备商业海报品质。
- 文本-图像符合度（1-5 分）：1=与提示词完全无关；3=部分相关但细节缺失；5=完美匹配提示词全部要素。

**客观评测指标**：
- CLIP Score：cos(e_I, e_T)，其中 e_I = CLIP_img(I), e_T = CLIP_text(T)。范围 [0,1]，越高表示语义匹配越好。
- 简化 FID：||μ_A - μ_B||² + Tr(Σ_A + Σ_B - 2(Σ_A·Σ_B)^{1/2})。以 CLIP 图像特征替代 InceptionV3，计算各组与 G1 基线的 Fréchet 距离。
- 美学启发式指标：H_aes = 1.5 + 2.0·ρ_edge + 1.0·σ_sat + 0.5·σ_val，范围 1-5。其中 ρ_edge 为 Canny 边缘像素占比，σ_sat/σ_val 为 HSV 饱和度/明度标准差。

### 3.4 对照实验设计

| 实验组 | ControlNet | IP-Adapter | CN 权重 | IP 权重 |
|--------|:---:|:---:|:---:|:---:|
| G1 (基线) | ✗ | ✗ | 0.0 | 0.0 |
| G2 (仅布局) | ✓ | ✗ | 0.18 | 0.0 |
| G3 (仅风格) | ✗ | ✓ | 0.0 | 0.33 |
| G4 (完整系统) | ✓ | ✓ | 0.18 | 0.33 |

实验固定 4 个测试主题：古风美人、青绿山水、赛博朋克、春日花卉，覆盖人物、风景、城市、自然四类视觉场景。每组使用相同随机种子 (seed=42)、推理步数 (25) 与 CFG Scale (7.5)，确保对比公平性。

### 3.5 预期结果

预期 G4（完整系统）在 CLIP Score 与主观美观度上均达到最优：ControlNet 弱约束提供构图引导，IP-Adapter 注入真实风格特征，两者互补而非冲突。G2 与 G3 分别在构图/风格维度上优于 G1 基线。

## 4 初步数据集与实验设想

### 4.1 评测数据集

构建包含 4 类主题、每类 4 组对照实验的评测数据集，共计 16 组生成实验：

| 主题 | 提示词要素 | 风格参考图特征 |
|------|----------|--------------|
| 古风美人 | 仕女、汉服、桃花、工笔、宋韵 | 暖色调古典人物画 |
| 青绿山水 | 山川、河流、松树、亭台、水墨 | 蓝绿色青绿山水画 |
| 赛博朋克 | 霓虹、雨夜、摩天楼、飞行器 | 紫青色城市夜景 |
| 春日花卉 | 樱花、郁金香、晨光、散景、微距 | 粉色调花卉摄影 |

### 4.2 算力预算与优化策略

| 项目 | 配置/策略 |
|------|---------|
| GPU | NVIDIA RTX 5060 Ti, 16 GB VRAM |
| 系统内存 | 32 GB DDR5 |
| 模型精度 | float16 (全流程) |
| 内存优化 | enable_model_cpu_offload, VAE slicing/tiling |
| 推理步数 | 25 步（评测）/ 15 步（调试） |
| 单张生成时间 | ≈15-25 秒 (1024×1024) |
| 完整评测周期 | ≈5-10 分钟 (16 组实验) |

### 4.3 时间规划

| 阶段 | 时间 | 主要任务 |
|------|------|---------|
| 第一阶段 | 第 1-2 周 | 环境搭建：PyTorch、Diffusers 安装，模型下载，HF 镜像配置 |
| 第二阶段 | 第 3-4 周 | 核心引擎开发：Engine 类实现，Gradio WebUI 搭建 |
| 第三阶段 | 第 5-6 周 | 多模态集成：ControlNet+IP-Adapter 联调，DeepSeek 提示词增强 |
| 第四阶段 | 第 7-8 周 | 评测实验：4 组对照实验全量运行，CLIP Score/FID 计算 |
| 第五阶段 | 第 9-10 周 | 论文撰写与答辩准备：开题报告定稿，PPT 制作 |

### 4.4 预期成果

（1）一套完整的图文驱动个性化海报/封面生成系统，包含核心引擎、Web 交互界面与评测框架。

（2）一份包含 4 组对照实验的定量评测报告，验证多模态控制的有效性。

（3）课程大作业终期报告（含系统设计、实验分析、可视化结果、结论）。

## 参考文献

[1] D. Podell, Z. English, K. Lacey, et al. "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis," arXiv:2307.01952, 2023.

[2] L. Zhang, A. Rao, and M. Agrawala. "Adding Conditional Control to Text-to-Image Diffusion Models," ICCV, 2023, pp. 3836-3847.

[3] H. Ye, J. Zhang, S. Liu, et al. "IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models," arXiv:2308.06721, 2023.

[4] B. Ai, Y. Zhou, Z. Chen, et al. "HCP-Diffusion: A Universal Stable-Diffusion Fine-tuning Framework," GitHub, 2024.

[5] Z. Yang, Y. Xu, and S. Liu. "Text2Poster: Laying Out Stylized Texts on Retrieved Images," ICASSP, 2022.

[6] Y. Li, J. Wang, et al. "PosterLayout: A New Benchmark and Approach for Content-Aware Visual-Textual Presentation Layout," CVPR, 2023.

[7] T. Karras, S. Laine, and T. Aila. "A Style-Based Generator Architecture for Generative Adversarial Networks," CVPR, 2019, pp. 4401-4410.

[8] R. Rombach, A. Blattmann, D. Lorenz, et al. "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR, 2022, pp. 10684-10695.

[9] C. Mou, X. Wang, L. Xie, et al. "T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models," AAAI, 2024.
