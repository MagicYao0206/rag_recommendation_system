# 1. 依赖导入层
import pandas as pd

import sys
import os
# 打印Python搜索路径
print("Python搜索路径：", sys.path)

# 尝试定位faiss安装位置
try:
    import faiss
    print("faiss安装路径：", faiss.__file__)
except ImportError:
    # 手动检查当前环境的site-packages中是否有faiss
    site_packages = [p for p in sys.path if 'site-packages' in p]
    print("环境site-packages路径：", site_packages)
    for sp in site_packages:
        faiss_path = os.path.join(sp, 'faiss')
        print(f"检查路径 {faiss_path} 是否存在：", os.path.exists(faiss_path))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

# 2. 资源加载层：加载预构建的向量库、映射表、嵌入模型（与构建层一致）
index = faiss.read_index(r"F:\rag_recommendation_system\vector_db\amazon_beauty_index.faiss")
chunks_df = pd.read_csv(r"F:\rag_recommendation_system\vector_db\chunks_mapping.csv", encoding="utf-8-sig")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. 核心函数层：封装检索逻辑（对外提供统一接口）
def translate_to_english(text):
    """将中文查询翻译成英文"""
    try:
        # 检测是否为中文，若是则翻译
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            translated = GoogleTranslator(source='zh-CN', target='en').translate(text)
            print(f"📌 中文查询已翻译：{text} → {translated}")
            return translated
        return text  # 非中文直接返回
    except Exception as e:
        print(f"⚠️ 翻译失败，使用原文本检索：{e}")
        return text
    
def retrieve_products(query, top_k=5):
    """
    输入用户提问，返回Top-K相关商品
    :param query: 用户提问（英文）
    :param top_k: 最终召回商品数
    :return: 去重后的商品DataFrame
    """
    query_english = translate_to_english(query)
    # 步骤1：生成提问的向量（与片段向量同模型/同归一化）
    query_embedding = model.encode([query_english], normalize_embeddings=True).astype("float32")
    
    # 步骤2：FAISS检索（多召回3倍，用于去重）
    scores, indices = index.search(query_embedding, top_k * 3)  # 如top_k=5，先召回15个片段
    
    # 步骤3：映射到文本片段，添加相似度得分
    retrieved_chunks = chunks_df.iloc[indices[0]]  # 按检索索引取片段
    retrieved_chunks = retrieved_chunks.copy()  # 先创建副本
    retrieved_chunks["similarity_score"] = scores[0]
    
    # 步骤4：按商品ID去重（一个商品可能对应多个片段，保留得分最高的）
    retrieved_products = retrieved_chunks.sort_values("similarity_score", ascending=False)
    retrieved_products = retrieved_products.drop_duplicates(subset="parent_asin", keep="first")
    
    # 步骤5：返回Top-K核心信息
    result = retrieved_products[["parent_asin", "title", "price", "main_category", "similarity_score"]].head(top_k)
    return result

# 4. 测试层：验证函数功能（独立运行时执行）
if __name__ == "__main__":
    query = "oil control facial cleanser for acne-prone skin"
    results = retrieve_products(query, top_k=3)
    print("🔍 检索结果：")
    print(results)