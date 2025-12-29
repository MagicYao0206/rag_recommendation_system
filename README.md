# 美妆商品RAG推荐系统

基于检索增强生成（RAG）技术的美妆商品智能推荐系统，结合向量检索与大语言模型，为用户提供精准且有针对性的美妆产品推荐。

## 项目简介

本项目构建了一个端到端的美妆商品推荐系统，通过以下流程实现智能推荐：
1. 从Amazon Beauty数据集提取商品信息与用户评论
2. 对文本数据进行预处理与片段切分
3. 构建商品向量数据库实现高效语义检索
4. 结合Llama-2大语言模型生成自然语言推荐理由
5. 提供Streamlit交互界面方便用户使用

## 项目结构

```
rag_recommendation_system
├─ app.py               # Streamlit交互界面
├─ build_vector_db.py   # 向量数据库构建脚本
├─ data                 # 数据存储目录
│  ├─ amazon_beauty_chunks.csv  # 切分后的文本片段
│  └─ amazon_beauty_rag.csv     # 预处理后的RAG数据集
├─ download_amazon.py   # Amazon数据集下载与处理脚本
├─ llm_recommend.py     # LLM推荐生成逻辑
├─ models               # 模型存储目录
│  └─ llama-2-7b-chat.Q4_K_M.gguf  # Llama-2模型文件
├─ requirements.txt     # 项目依赖
├─ retrieval.py         # 检索功能实现
├─ text_preprocess.py   # 文本预处理与切分
└─ vector_db            # 向量数据库存储
   ├─ amazon_beauty_index.faiss  # FAISS向量索引
   └─ chunks_mapping.csv         # 向量-文本映射表
```

## 环境要求

- Python 3.8+
- 推荐8GB以上内存

## 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/MagicYao/rag_recommendation_system.git
cd rag_recommendation_system
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 准备模型文件
   - 下载Llama-2-7B-Chat-GGUF模型（如llama-2-7b-chat.Q4_K_M.gguf）
   - 放置到`models`目录下

## 使用指南

### 1. 数据准备（首次使用时）

```bash
# 下载并处理Amazon Beauty数据集
python download_amazon.py

# 文本预处理与切分
python text_preprocess.py

# 构建向量数据库
python build_vector_db.py
```

### 2. 启动推荐系统

```bash
streamlit run app.py
```

3. 在浏览器中访问显示的本地地址（通常为http://localhost:8501）
4. 输入美妆需求（例如"适合敏感肌的面霜"）
5. 点击"生成推荐"按钮获取推荐结果

## 核心功能

- **智能检索**：基于Sentence-BERT的向量检索，精准匹配用户需求
- **多语言支持**：自动将中文查询翻译为英文进行检索
- **本地部署**：所有模型和数据均在本地运行，保护用户隐私
- **流式输出**：推荐理由实时生成并显示，提升用户体验

## 技术细节

- **检索模块**：使用Sentence-BERT (all-MiniLM-L6-v2)生成文本向量，FAISS实现高效近似最近邻搜索
- **LLM模块**：采用Llama-2-7B-Chat大语言模型，通过llama.cpp库实现本地部署
- **数据处理**：使用Pandas进行数据处理，LangChain进行文本切分
- **前端交互**：基于Streamlit构建简洁直观的用户界面

## 注意事项

- 模型文件较大（约4GB），请确保有足够的存储空间
- 首次运行时数据下载和向量库构建可能需要较长时间
- 可根据硬件配置调整`llm_recommend.py`中的模型参数（如n_gpu_layers、n_threads）
