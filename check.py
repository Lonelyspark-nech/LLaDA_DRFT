from transformers import AutoTokenizer
import torch
import pathlib
import sys
from pathlib import Path
# 获取当前文件所在目录（Bitnet-main）
current_dir = Path(__file__).resolve().parent

# 添加 LLaDA 模型源码路径（modeling_llada.py 所在路径）
llada_path = current_dir.parent / "hf_cache" / "hub" / "models--GSAI-ML--LLaDA-8B-Base" / "snapshots" / "ce71e3c2523f535e022bccedbda192eb8869fd44"  # ← 替换完整hash路径

sys.path.insert(0, str(llada_path.resolve()))

# 现在可以导入
from modeling_llada import LLaDAModelLM


# 加载模型和 tokenizer
path = "/root/autodl-tmp/hf_cache/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/ce71e3c2523f535e022bccedbda192eb8869fd44"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = LLaDAModelLM.from_pretrained(path)
model.eval()

# 输入示例
inputs = tokenizer("hello world", return_tensors="pt")

# 调用 encoder-only 并启用 hidden_states
with torch.no_grad():
    outputs = model.model(**inputs, output_hidden_states=True)

# ✅ 获取最后一层隐藏状态（LN之后）
hidden = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]

# ✅ 使用 mean pooling / eos pooling 提取句向量
# 方式一：mean pooling（适合 dense retrieval）
attention_mask = inputs['attention_mask']
masked_hidden = hidden * attention_mask.unsqueeze(-1)  # [B, T, D]
sentence_embeddings = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

print("Embedding shape:", sentence_embeddings.shape)
