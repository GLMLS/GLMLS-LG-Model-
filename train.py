# 导入必要的库
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict
from torch.utils.data import Dataset, DataLoader

# 导入GLMLS_LG模型库
from GLMLS_LG_Model import GLMLS_LG, LG_ModelConfig as ModelConfig  # 正确导入配置类
from GLMLS_LG_Model import LG_DynamicVocabulary as DynamicVocabulary  # 正确导入动态词汇表类

# 加载配置文件
def load_config(config_path: str) -> Dict:
    """从指定路径加载JSON配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 创建模型实例
def create_model(config: Dict, mode: str) -> GLMLS_LG:
    """根据配置创建模型实例，支持从头训练或继续训练"""
    model_config = ModelConfig(
        block_size=config['model']['block_size'],
        n_layer=config['model']['n_layer'],
        n_head=config['model']['n_head'],
        n_embed=config['model']['n_embed'],
        vocab_size=config['model']['vocab_size'],
        dropout=config['model']['dropout']
    )
    
    model = GLMLS_LG(model_config)
    
    if mode == 'resume':
        # 实现继续训练逻辑
        checkpoint_path = os.path.join(config['training']['checkpoint_dir'], 'latest.pth')
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
        else:
            print("未找到检查点文件，将从头开始训练")
    
    return model

# 自定义的文本数据集类
class TextDataset(Dataset):
    """用于处理文本数据的数据集类"""
    def __init__(self, text: str, block_size: int, vocab):
        self.text = text
        self.block_size = block_size
        self.vocab = vocab
        
    def __len__(self):
        return len(self.text) // self.block_size
    
    def __getitem__(self, idx):
        # 获取数据块
        start = idx * self.block_size
        end = start + self.block_size
        chunk = self.text[start:end]
        
        # 将文本编码为整数序列
        inputs = self.vocab.encode(chunk)
        # 对于语言模型任务，输入和标签是相同的序列
        labels = inputs.copy()
        
        # 转换为PyTorch张量
        inputs_tensor = torch.tensor(inputs, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return {"input_ids": inputs_tensor, "labels": labels_tensor}

# 训练主函数
def train(config_path: str):
    """主训练函数，实现完整的模型训练流程"""
    # 加载配置文件
    config = load_config(config_path)
    
    # 加载训练数据
    with open(config['training']['data_path'], 'r', encoding='utf-8') as f:
        train_text = f.read()
    
    # 创建动态词汇表
    vocab = DynamicVocabulary()
    vocab.build_from_text(train_text)
    
    # 创建数据集和数据加载器
    dataset = TextDataset(train_text, config['model']['block_size'], vocab)
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
    
    # 创建模型
    train_mode = config['training']['mode']
    model = create_model(config, train_mode)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    # 检查是否有GPU可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 训练循环
    for epoch in range(config['training']['num_epochs']):
        model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(input_ids)
            
            # 计算损失
            loss = criterion(outputs.view(-1, config['model']['vocab_size']), labels.view(-1))
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % config['training']['log_interval'] == 0:
                print(f"Epoch [{epoch+1}/{config['training']['num_epochs']}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{config['training']['num_epochs']}], Average Loss: {avg_loss:.4f}")
        
        # 每个epoch保存检查点
        checkpoint_dir = config['training']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
    
    # 最后保存最终模型
    final_checkpoint_path = os.path.join(checkpoint_dir, 'latest.pth')
    torch.save(model.state_dict(), final_checkpoint_path)
    print("训练完成，模型已保存")

# 当脚本直接运行时，启动训练
if __name__ == "__main__":
    train('config.json')
