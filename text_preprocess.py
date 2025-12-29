# 1. 依赖导入层：导入所需库（数据处理+文本切分）
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 2. 数据加载层：读取清洗后的核心数据集
df = pd.read_csv(r"F:\rag_recommendation_system\data\amazon_beauty_rag.csv", encoding="utf-8-sig")

# 3. 工具初始化层：配置文本切分器（核心参数：片段长度/重叠/分隔符）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,        # 每个片段300字符（英文适配）
    chunk_overlap=50,       # 片段重叠50字符（避免语义割裂）
    separators=["\n", ". ", " ", ""],  # 切分优先级（先按换行，再按句点）
)

# 4. 核心处理层：遍历数据，切分文本并保留元信息
chunks = []
for idx, row in df.iterrows():
    if pd.notna(row["rag_text"]):  # 过滤空值
        text_chunks = text_splitter.split_text(row["rag_text"])  # 切分文本
        for chunk in text_chunks:
            chunks.append({
                "parent_asin": row["parent_asin"],  # 商品ID（关联主键）
                "title": row["title"],              # 商品标题
                "price": row["price"],              # 价格
                "main_category": row["main_category"],  # 类目
                "chunk_text": chunk,                # 切分后的短文本（核心）
                "full_rag_text": row["rag_text"]    # 完整文本（备用）
            })

# 5. 结果保存层：将切分后的片段存入CSV，供后续向量生成使用
chunks_df = pd.DataFrame(chunks)
chunks_df.to_csv(r"F:\rag_recommendation_system\data\amazon_beauty_chunks.csv", index=False, encoding="utf-8-sig")

# 6. 日志输出层：提示处理完成，便于验证
print(f"✅ 文本切分完成！生成 {len(chunks_df)} 个文本片段")