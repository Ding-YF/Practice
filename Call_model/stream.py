import os
import dashscope

messages = [
    {'role': 'user', 'content': '你好~'}
]

response = dashscope.Generation.call(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    model="deepseek-r1",  # 此处以deepseek-r1为例，可按需更换模型名称。
    messages=messages,
    result_format='message',
    stream=True,
    incremental_output=True
)

# 定义完整思考过程
reasoning_content = ""
# 定义完整回复
answer_content = ""
# 判断是否结束思考过程并开始回复
is_answering = False

print("=" * 20 + "思考过程" + "=" * 20)

for chunk in response:
    # 如果思考过程与回复皆为空，则忽略
    if (chunk.output.choices[0].message.content == "" and 
        chunk.output.choices[0].message.reasoning_content == ""):
        pass
    else:
        # 如果当前为思考过程
        if (chunk.output.choices[0].message.reasoning_content != "" and 
            chunk.output.choices[0].message.content == ""):
            print(chunk.output.choices[0].message.reasoning_content, end="")
            reasoning_content += chunk.output.choices[0].message.reasoning_content
        # 如果当前为回复
        elif chunk.output.choices[0].message.content != "":
            if not is_answering:
                print("\n" + "=" * 20 + "完整回复" + "=" * 20)
                is_answering = True
            print("-"*5)
            print(chunk.output.choices[0].message.content, end="")
            answer_content += chunk.output.choices[0].message.content

# 如果您需要打印完整思考过程与完整回复，请将以下代码解除注释后运行
# print("=" * 20 + "完整思考过程" + "=" * 20 + "\n")
# print(f"{reasoning_content}")
# print("=" * 20 + "完整回复" + "=" * 20 + "\n")
# print(f"{answer_content}")