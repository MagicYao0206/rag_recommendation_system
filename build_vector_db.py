# 1. 依赖导入层：数据处理+向量库+数值计算+嵌入模型
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 2. 数据加载层：读取切分后的文本片段
chunks_df = pd.read_csv(r"F:\rag_recommendation_system\data\amazon_beauty_chunks.csv", encoding="utf-8-sig")

# 3. 模型加载层：选择轻量英文嵌入模型（all-MiniLM-L6-v2：384维，速度快）
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. 向量生成层：批量编码文本（避免一次性加载过多数据导致内存溢出）
batch_size = 1000  # 批量大小（根据内存调整，8G内存建议500）
embeddings = []
for i in range(0, len(chunks_df), batch_size):
    batch_texts = chunks_df["chunk_text"].iloc[i:i+batch_size].tolist()
    batch_embeddings = model.encode(batch_texts, normalize_embeddings=True)  # 归一化（适配余弦相似度）
    embeddings.extend(batch_embeddings)
embeddings = np.array(embeddings).astype("float32")  # FAISS要求float32类型

# 5. 向量库构建层：初始化FAISS索引并添加向量
dimension = embeddings.shape[1]  # 获取向量维度（384）
index = faiss.IndexFlatIP(dimension)  # IndexFlatIP：内积检索（归一化后=余弦相似度）
index.add(embeddings)  # 将所有向量加入索引

# 6. 结果保存层：保存向量库+映射表（缺一不可！）
faiss.write_index(index, r"F:\rag_recommendation_system\vector_db\amazon_beauty_index.faiss")
chunks_df.to_csv(r"F:\rag_recommendation_system\vector_db\chunks_mapping.csv", index=False, encoding="utf-8-sig")

# 7. 日志输出层：验证构建结果
print(f"✅ 向量库构建完成！")
print(f"📊 向量维度：{dimension}，片段数：{len(chunks_df)}")