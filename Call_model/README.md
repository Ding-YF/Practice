## 说明
> 下述模型通过阿里百炼平台调用

GENERAL_MODELS = [
    'qwen-max-latest', 'llama3.1-8b-instruct', 'llama3.1-70b-instruct', 'llama3.1-405b-instruct',
    'deepseek-r1','deepseek-v3'
]

CODER_MODELS = ['qwen-coder-turbo-latest']

VISUAL_MODELS = ['qwen-vl-max']

## 使用方法 
1. 参考阿里[官方教程](https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key?spm=a2c4g.11186623.help-menu-2400256.d_3_0.628a47bbdbZDBL)
   1. 获取API-KEY
   2. 配置KEY到环境变量
   3. 安装`dashcope`SDK
2. 运行 `dashcope_sdk.py`
3. 根据提示选择模型种类(输入序号选择)
   1. 对话模型：依据提示选择需要使用的模型
   2. 编程模型：先输入编程目标，再输入编程要求
   3. 视觉模型：先输入图片地址，再输入要求
4. 对话记录会自动保存在log文件夹下，以时间命名(仅对话模型)

## 注意
- 其中通用模型的输出进行了优化，包括Markdwon渲染(默认开启)和流式输出(仅支持deepseek-r1,需要手动开启);
- LLM-Launch.bat 是 windows 下的启动脚本，根据情况修改程序路径，放入`Start Menu`后，配合PowerToys可以方便快速的启动程序;