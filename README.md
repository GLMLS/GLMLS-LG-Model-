# GLMLS_LG Model README

# 简介

GLMLS-LG龙光（GLMLS Long Guang Large Language Model）是由金龙华光工作室的GoldenLoongAS开发的轻量级高效语言模型，专注于平衡性能与资源消耗，适合在各类硬件环境下运行。


# 特点


• 高效架构：采用优化的Transformer结构，在保持性能的同时降低资源需求

• 动态适应：支持动态词汇表和认知窗口机制，提升长文本处理效率

• 多平台支持：兼容CPU和GPU硬件环境

• 灵活生成：提供多种文本生成策略，包括Top-k、Top-p采样和束搜索


# 安装指南


通过PyPI安装

Linux系统（推荐aarch64架构）：


```bash
# 创建并激活虚拟环境（可选但推荐）
python3 -m venv glmls-env
source glmls-env/bin/activate

# 安装GLMLS_LG模型轮子文件
pip install GLMLS_LG_Model-0.0.1-cp311-cp311-linux_aarch64.whl

```


Windows系统（兼容模式）：


```cmd
REM 创建并激活虚拟环境（可选但推荐）
python -m venv glmls-env
glmls-env\Scripts\activate

REM 安装GLMLS_LG模型轮子文件
pip install GLMLS_LG_Model-0.0.1-cp311-cp311-linux_aarch64.whl

```

你也可以在本项目main分支中下载`GLMLS_LG_Model-0.0.1-cp311-cp311-linux_aarch64.whl`后,使用
```bash
# 安装GLMLS_LG模型轮子文件
pip install GLMLS_LG_Model-0.0.1-cp311-cp311-linux_aarch64.whl
```
命令安装。

# 模型架构图解


```
文本输入 --> [编码层] --> [动态窗口处理]
                      |
                      v
              [Transformer Block]
              (含Flash Attention)
                      |
                      v
              [前馈神经网络]
                      |
                      v
              [输出层 & 解码]
                      |
                      v
                生成文本输出
```



# 使用示例


```
# 导入必要的库
import torch
import os
import numpy as np
from GLMLS_LG_model import LG_ModelConfig, LG_DynamicVocabulary, GLMLS_LG

# 初始化配置和模型
# 使用LG_ModelConfig类来创建模型配置实例
config = LG_ModelConfig()
# 初始化动态词汇表管理类
vocab = LG_DynamicVocabulary()
# 根据配置创建GLMLS_LG模型实例
model = GLMLS_LG(config)

# 构建词汇表（从示例文本中提取字符并构建词汇表）
sample_text = "你好，欢迎使用GLMLS_LG模型！"
vocab.build_from_text(sample_text)

# 编码输入文本（将文本转换为模型可处理的整数序列）
input_ids = vocab.encode(sample_text)
# 将整数序列转换为PyTorch张量
input_tensor = torch.tensor([input_ids], dtype=torch.long)

# 生成文本（调用模型的文本生成方法）
output_ids = model.generate(input_tensor, max_new_tokens=50)
# 将生成的整数序列解码为文本
generated_text = vocab.decode(output_ids[0].tolist())

print("生成结果:", generated_text)

```



# 支持与反馈


• 官方网站：[金龙华光工作室](glmls.mxhmcp.sbs)

• 联系邮箱：glmls@mxhmcp.sbs

• 模型版本：0.0.1

• 支持硬件：CPU 和 GPU

• 许可证：Mozilla Public License 2.0(MPL 2.0)


# 许可证

本模型遵循Mozilla Public License 2.0(MPL 2.0)开源协议，允许自由使用、修改和分发，但需遵守协议规定的条款和条件，自由使用、修改和分发时必须注明版权著作权以及最终解释权归原作者。该模型受版权保护，仅限于非商业用途，如需用于商业用途请联系版权方获取授权。使用本模型即表示您同意遵守这些条款。