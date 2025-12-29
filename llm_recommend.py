import os
import sys
import pandas as pd
from retrieval import retrieve_products  

# 全局禁用外网请求（保持离线）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def init_llama_model():
    """初始化Llama-2模型（使用llama.cpp）"""
    try:
        from llama_cpp import Llama
        
        # 本地模型路径（请替换为你的实际路径）
        model_path = r"F:\rag_recommendation_system\models\llama-2-7b-chat.Q4_K_M.gguf"
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在：{model_path}")

        # 初始化Llama模型（配置参数根据硬件调整）
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,          # 上下文窗口大小
            n_threads=4,         # 线程数（根据CPU核心数调整）
            n_gpu_layers=20,     # GPU加速层数（0表示仅用CPU）
            verbose=False        # 关闭详细日志
        )
        
        print("✅ Llama模型加载成功")
        return llm
    
    except Exception as e:
        print(f"❌ 模型加载失败：{str(e)}")
        return None

def generate_recommendation_stream(retrieved_data, llm):
    """基于Llama模型流式生成推荐结果"""
    # 构造商品检索文本（保持不变）
    product_text = "\n".join([
        f"{i+1}. {row['title']}（相似度{row['similarity_score']:.2f}）"
        for i, row in retrieved_data.iterrows()
    ])
    
    # 提示词模板（保持不变）
    prompt = f"""[INST] <<SYS>>
    你是专业的美妆商品推荐师，基于提供的检索结果生成简洁准确的推荐理由。
    <</SYS>>

    基于以下检索结果生成3条推荐理由，每条不超过50字，带编号且单独占一行，仅输出推荐内容：
    {product_text} [/INST]"""
    
    # 流式生成配置
    stream = llm(
        prompt=prompt,
        max_tokens=300,
        temperature=0.7,
        stop=["</s>"],
        echo=False,
        stream=True  # 启用流式输出
    )
    
    # 逐 token 输出结果
    response = []
    for chunk in stream:
        token = chunk["choices"][0]["text"]
        response.append(token)
        print(token, end="", flush=True)  # 实时刷新输出
    print()  # 生成结束后换行
    return "".join(response)

# 测试主流程
if __name__ == "__main__":
    query = "适合敏感肌的面霜"
    retrieved_products = retrieve_products(query, top_k=3)
    llm = init_llama_model()
    if llm:
        print("\n🎯 商品推荐理由：")
        print(generate_recommendation_stream(retrieved_products, llm))
        llm.close()  # 显式关闭模型