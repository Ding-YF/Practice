## 说明
> 能够在windows种conda的rfuniverse环境中调用下列模型

> 下述模型通过阿里百炼平台调用

GENERAL_MODELS = [
    'qwen-max-latest', 'llama3.1-8b-instruct', 'llama3.1-70b-instruct', 'llama3.1-405b-instruct'
    'deepseek-r1','deepseek-v3'
]

CODER_MODELS = ['qwen-coder-turbo-latest']

VISUAL_MODELS = ['qwen-vl-max']

## 使用方法
1. 运行 `dashcope_sdk.py`
2. 根据提示选择模型种类
   1. 对话模型：依据提示选择需要使用的模型
   2. 编程模型：先输入编程目标，在输入编程要求
   3. 视觉模型：先输入图片地址，在输入要求
3. 对话记录会自动保存在log文件夹下，以时间命名

## 注意
- 其中通用模型的输出进行了优化，包括Markdwon渲染(默认开启)和流式输出(仅支持deepseek-ri,需要手动开启);