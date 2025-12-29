# 1. 依赖导入层
import streamlit as st
from llm_recommend import generate_recommendation_stream, init_llama_model, retrieve_products

# 2. 页面配置层
st.set_page_config(
    page_title="美妆商品RAG推荐系统",
    page_icon="🎨",
    layout="wide"
)

# 3. 页面UI层
st.title("🎨 美妆商品智能推荐系统")
st.subheader("输入你的美妆需求，为你精准推荐商品:)")

query = st.text_input(
    "请输入需求（例如：控油洗面奶、适合敏感肌的面霜）：",
    placeholder="适合痘痘肌的控油洗面奶"
)

# 初始化Llama模型（替换原Qwen模型初始化）
llm = init_llama_model()
if not llm:
    st.error("❌ 模型加载失败，请检查模型路径是否正确")

if st.button("生成推荐", type="primary"):
    if not query:
        st.warning("⚠️ 请输入需求关键词！")
    elif not llm:
        st.error("❌ 模型未加载成功，无法生成推荐")
    else:
        with st.spinner("🤖 正在检索并生成推荐..."):
            retrieved_products = retrieve_products(query, top_k=3)
            recommendation = generate_recommendation_stream(retrieved_products, llm)
        st.success("✅ 推荐完成！")
        st.markdown("### 🌟 推荐结果：")
        st.markdown(recommendation)