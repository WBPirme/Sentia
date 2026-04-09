# Sentia - 本地运行的 AI 虚拟生命体桌面交互系统

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![llama.cpp](https://img.shields.io/badge/LLM-llama.cpp-green.svg)](https://github.com/ggerganov/llama.cpp)
[![VTube Studio](https://img.shields.io/badge/Live2D-VTube_Studio-pink.svg)](https://denchisoft.com/)
[![Local AI](https://img.shields.io/badge/Deploy-Local_Offline-orange.svg)]()

> **一句话总结**：Sentia 不是单纯的“语音聊天机器人”，而是一个将**本地大模型、长期记忆、情绪状态机和虚拟形象驱动**绑定在一起的虚拟角色系统原型。它的核心在于让角色“像真正活着一样，会记仇、会等待、会生气、会原谅”。

Sentia 是一个以“本地运行的虚拟生命体”为目标的桌面交互项目。它将大语言模型、语音识别、语音合成、长期记忆和 VTube Studio 角色驱动串联成一条完整的业务链路，支持键盘与麦克风双输入，打造一个常驻在桌面的陪伴式 AI 虚拟主播/角色。

## 核心特色

* ** 本地大模型驱动**：基于 `llama.cpp`，支持根据本机显卡环境（CUDA/HIP/Vulkan/CPU）自动切换引擎目录，提供极低延迟的本地推理。
* ** 融合双态输入**：支持终端键盘打字与“按住空格键”流式语音输入。同轮对话中，两者竞速提交，带来极其自然的交互体验。
* ** 纯本地听说能力**：基于 `sherpa-onnx` 实现流式语音识别（ASR）与 VITS 语音合成（TTS），音频直接本地播放并同步驱动角色口型。
* ** 双模长期记忆系统**：不仅记录对话，还能记录“强烈情绪”、“关机”等高优事件。支持本地向量数据库（基于 ChromaDB）检索，缺失嵌入模型时自动降级为轻量文本记忆。
* ** 情绪与动作决策引擎**：模型输出结构化数据（包含 `text`、`emotion`、`action`、`patience`）。角色不仅会说话，还会决定当前情绪（中立/微笑/生气）以及行为（如主动关机、原谅用户）。
* ** 动态忍耐度与幽灵模式**：角色会根据情绪决定等待用户回复的时间。多次被无视后可能触发 `Shutdown` 进入幽灵模式，关闭 VTS 身体，仅接受用户的“语音道歉”以决定是否 `Forgive`。
* ** VTube Studio 深度联动**：通过 `pyvts` 持续注入参数，控制角色的头部转动、眼神注视、眨眼、呼吸感与表情，赋予角色真实的“生命体征”。

---

## 系统架构与核心机制

### 核心处理循环
1. 等待用户输入（键盘或语音超时监控）。
2. 结合“被无视次数”、“幽灵模式状态”以及“历史记忆检索”构造本轮 Prompt。
3. 提交至本地 LLM，解析返回的结构化指令 (`text` / `emotion` / `action` / `patience`)。
4. 触发 TTS 生成语音，同步驱动 VTube Studio 口型与表情参数。
5. 执行动作分支（正常对话 / 关机 / 拒绝 / 原谅），并将关键事件异步写入长期记忆。

### 技术栈构成
* **主控逻辑**: `Python`
* **大语言模型**: `llama.cpp` (LLM 引擎), `OpenAI Python SDK` (接口通信)
* **语音交互**: `sherpa-onnx` (ASR/TTS), `sounddevice` (音频控制)
* **记忆系统**: `chromadb` (向量存储), `sentence-transformers` (文本嵌入)
* **视觉呈现**: `pyvts` (VTS 交互)
* **硬件交互**: `keyboard` (监听空格键)

---

## 快速开始与部署指南

本项目推荐在 **Windows + Python 3.12** 环境下运行，并已安装 [VTube Studio](https://store.steampowered.com/app/1325860/VTube_Studio/)。

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 准备推理引擎 (llama.cpp)
在项目根目录创建 `engine/` 文件夹，并放入对应环境的 `llama.cpp` 预编译包（需包含 `llama-server.exe` 及相关 DLL）：
```text
engine/
 ├─ llama.cpp-cpu/
 ├─ llama.cpp-cuda12/
 ├─ llama.cpp-hip/
 └─ llama.cpp-vulkan/
```

### 3. 下载并配置模型资源
在根目录创建 `models/` 文件夹，按照以下结构存放对应模型：

| 模型类型 | 推荐下载链接/来源 | 目标存放路径 |
| :--- | :--- | :--- |
| **LLM (GGUF)** | [Sentia-Qwen3.5-9B](https://huggingface.co/BucketP/Sentia-Qwen3.5-9B-GGUF) | `models/Sentia-Q4_K_M.gguf` 或 `Sentia-9B-FP16.gguf` |
| **TTS (语音合成)** | [GitHub Releases](https://github.com/WBPirme/Sentia_VTuber_Agent/releases/tag/v1.0.0) | `models/G_28300.onnx` 及 `tokens.txt` |
| **ASR (语音识别)** | [GitHub Releases](https://github.com/WBPirme/Sentia_VTuber_Agent/releases/tag/v1.0.0) | `models/asr/` (解压至此) |
| **记忆数据库** | [GitHub Releases](https://github.com/WBPirme/Sentia_VTuber_Agent/releases/tag/v1.0.0) | `models/memory_db/` (初始库，解压至此) |
| **Embedding 模型** | [bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) / [GitHub Releases](https://github.com/WBPirme/Sentia_VTuber_Agent/releases/tag/v1.0.0) | `models/memory_embedding/` (包含 config, model 等文件) |

*(注：如果你希望使用作者提供的 Live2D 模型，可在 Releases 中下载 `VTuber_model` 并导入 VTube Studio)*

### 4. 调整配置文件
打开 `main.py`，找到并修改 `VTS_EXE_PATH` 为你本机 VTube Studio 的实际安装路径：
```python
VTS_EXE_PATH = r"E:\SteamLibrary\steamapps\common\VTube Studio\VTube Studio.exe"
```

### 5. 启动系统
```bash
python main.py
```
*(系统启动后，会提示选择模型并在后台异步拉起服务、初始化各引擎组件，最终进入等待输入状态。输入 `exit` 或 `quit` 可退出程序。)*

---

## 记忆管理工具 (Memory Admin)

项目内置了独立的 CLI 工具用于管理角色的长期记忆库：

```bash
# 查看帮助与支持的命令
python tools/memory_admin.py --help

# 常用操作示例：
python tools/memory_admin.py list --limit 20                  # 查看最近 20 条记忆
python tools/memory_admin.py search "你还记得什么" --limit 5     # 语义检索
python tools/memory_admin.py add "用户很在意我是否记得。" --emotion Smile --importance 4 # 手动添加
python tools/memory_admin.py prune --contains 关机 --dry-run  # 批量清理测试
```

---

## 目录结构

```text
Sentia/
├─ main.py                 # 程序主入口与交互主循环
├─ core/
│  ├─ asr_engine.py        # 流式语音识别
│  ├─ llm_controller.py    # 引擎启停与硬件路由
│  ├─ memory_engine.py     # 向量/轻量双模记忆引擎
│  ├─ tts_engine.py        # 文本转语音与重采样
│  └─ vts_controller.py    # VTS API 通信与动作映射
├─ tools/
│  └─ memory_admin.py      # 记忆管理 CLI 工具
├─ assets/                 # 词典与发音基础资源
└─ models/                 # 所有大模型、小模型及数据库文件目录
```

---

## ⚠️ 已知限制与说明

* **平台兼容性**：当前深度绑定 Windows 环境（涉及键盘钩子与特定路径）。
* **硬件依赖**：显卡后端路由依赖本机环境检测，若自动失败需手动整理 `engine/` 目录；嵌入模型（Embedding）缺失时自动降级但会影响记忆检索精度。
* **角色表现**：角色性格与行为高度绑定于 Prompt 设定和 GGUF 模型能力，更换不同基座模型可能会导致输出格式不稳定或“人设崩塌”。
* **不含 VTS 的降级体验**：如果没有 VTube Studio，视觉身体层级将缺失，但文本、语音、记忆及情绪逻辑核心依然可独立运行。

---

*“让 AI 不再仅仅是服务者，而是陪在你桌面的、有情绪、有记忆的生命。”*
```
