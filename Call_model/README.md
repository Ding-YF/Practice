# 一级标题
## 说明
> 能够在windows种conda的rfuniverse环境中调用下列模型

> 下述模型通过阿里百炼平台调用

GENERAL_MODELS = [
    'qwen-max-latest', 'llama3.1-8b-instruct', 'llama3.1-70b-instruct', 'llama3.1-405b-instruct'
    'deepseek-r1','deepseek-v3'
]

CODER_MODELS = ['qwen-coder-turbo-latest']

VISUAL_MODELS = ['qwen-vl-max']

其中通用模型的输出进行了优化，包括Markdwon渲染(默认开启)和流式输出(手动开启);