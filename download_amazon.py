import os
import pandas as pd
from datasets import load_dataset

# ========== 核心配置 ==========
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
TARGET_DIR = r"F:\rag_recommendation_system\data"
REVIEW_SUBSET = "raw_review_All_Beauty"
META_SUBSET = "raw_meta_All_Beauty"
SAMPLE_SIZE = 500

# ========== 确保目录存在 ==========
os.makedirs(TARGET_DIR, exist_ok=True)

# ========== 加载数据集 + 打印字段（确认真实字段名） ==========
def load_and_check_fields(subset_name):
    try:
        dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            name=subset_name,
            split="full",
            trust_remote_code=True,
            revision="main"
        )
        df = dataset.to_pandas()
        print(f"\n✅ 加载 {subset_name} 成功！字段名列表：")
        print(df.columns.tolist())
        return df
    except Exception as e:
        print(f"❌ 加载 {subset_name} 失败：{e}")
        return None

print("开始加载评论数据...")
df_reviews = load_and_check_fields(REVIEW_SUBSET)
print("\n开始加载商品元数据...")
df_meta = load_and_check_fields(META_SUBSET)

if df_reviews is None or df_meta is None:
    print("\n❌ 数据集加载失败！")
    exit()

# ========== 适配真实字段名 ==========
# 1. 评论数据：用parent_asin作为商品ID（和元数据对齐），text作为评论内容
df_reviews_clean = df_reviews[["parent_asin", "text"]].head(SAMPLE_SIZE)
df_reviews_clean.rename(columns={"text": "reviewText"}, inplace=True)  # 统一字段名

# 2. 商品元数据：用parent_asin去重，保留核心字段
meta_fields = ["parent_asin", "title", "description", "price", "main_category"]  # 修正：main_category是类目字段
df_meta_clean = df_meta[meta_fields].drop_duplicates(subset="parent_asin")  # 修正：用parent_asin去重

# ========== 合并商品+评论（用parent_asin对齐） ==========
df_merged = pd.merge(
    df_meta_clean,
    df_reviews_clean.groupby("parent_asin")["reviewText"].apply(lambda x: " | ".join(x[:3])).reset_index(),
    on="parent_asin",
    how="left"
)

# ========== 填充空值 + 生成RAG检索文本 ==========
# 填充缺失值
df_merged["title"] = df_merged["title"].fillna("No title")
df_merged["description"] = df_merged["description"].fillna("No description")
df_merged["price"] = df_merged["price"].fillna("0.0")
df_merged["main_category"] = df_merged["main_category"].fillna("Unknown category")
df_merged["reviewText"] = df_merged["reviewText"].fillna("No user reviews")

# 生成RAG核心文本（标题+描述+类目+评论）
df_merged["rag_text"] = (
    "Title: " + df_merged["title"] + ". " +
    "Description: " + df_merged["description"] + ". " +
    "Category: " + df_merged["main_category"] + ". " +
    "Price: " + df_merged["price"] + ". " +
    "User reviews: " + df_merged["reviewText"]
)

# ========== 导出到本地（最终可用的RAG数据集） ==========
csv_path = os.path.join(TARGET_DIR, "amazon_beauty_rag.csv")
df_merged.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig"  # Windows兼容编码
)

# ========== 输出成功日志 ==========
print(f"\n🎉 数据集处理完成！")
print(f"📂 最终文件：{csv_path}")
print(f"📊 有效商品数：{len(df_merged)} 条")
print(f"🔑 核心字段：{df_merged.columns.tolist()}")
print(f"\n💡 后续RAG开发可直接使用 'rag_text' 字段做语义检索！")