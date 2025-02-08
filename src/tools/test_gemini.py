import os

import openai
from dotenv import load_dotenv
import json

# 加载环境变量
env_path = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), '.env')
load_dotenv(env_path)

# 获取配置
api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL")
base_url = os.getenv("OPENAI_BASE_URL")


def test_simple_prompt():
    """测试简单的提示词生成"""
    print(f"Using model: {model_name}")

    # 初始化客户端
    client = openai.OpenAI(base_url=base_url, api_key=api_key)
    messages = [{"role": "user", "content": "Write a story about a magic backpack."}]
    # 测试简单生成
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.6,
    )

    print("\nSimple prompt response:")
    print("Response type:", type(response))
    print("Response attributes:", dir(response))

    # 打印完整的响应对象结构
    print("\nFull response structure:")
    print(json.dumps(response.model_dump(), indent=2))


if __name__ == "__main__":
    print("Testing Openai API...")
    test_simple_prompt()
    print("\n" + "=" * 50 + "\n")
