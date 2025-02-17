import os
# For loading animation
from rich.console import Console
from rich.markdown import Markdown
# For mulit input
from prompt_toolkit import PromptSession
# Model API
from dashscope import Generation
import dashscope
import datetime

GENERAL_MODELS = [
    'qwen-max-latest', 'llama3.1-8b-instruct', 'llama3.1-70b-instruct', 'llama3.1-405b-instruct',
    'deepseek-r1','deepseek-v3'
]
CODER_MODELS = ['qwen-coder-turbo-latest']
VISUAL_MODELS = ['qwen-vl-max']

filename = datetime.datetime.now().strftime("%Y%m%d_%H%M") + ".md"
if not os.path.exists("log"):
    os.mkdir("log")
file_path = os.path.join("log",filename)
def log(content):
    """Add dialog to markdown file"""
    try:
        with open(file_path,"a",encoding="utf-8") as file:
            file.write(f"{content}\n\n") #Markdown need double \n
        console = Console()
        console.print(f"[italic bright_black]Appended to {filename}[/]")
    except Exception as e:
        console.print(f"[italic red]write failed: {str(e)}[/]")

# Call general dialog model
def general_model():
    """Call the general dialog model."""
    os.system('clear')
    content = ""
    round_count = 1
    stream_output = False
    reason_switch = False
    answer_switch = False
    model_called = GENERAL_MODELS[4]
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
    while True:
        console = Console()
        console.print("[bold red]-[/]"*40+f"[bold red]第{round_count}轮对话[/]"+"[bold red]-[/]"*40)
        # Multi input
        session = PromptSession()
        in_content = session.prompt(
            console.print("[bold blue]用户输入(按Esc+Enter提交): [/]"),
            multiline = True,
            vi_mode=True
        )
        log(f"用户:{in_content}")
        # Append user input to messages
        messages.append({'role': 'user', 'content': in_content})
        # Loading Animation
        with console.status(
            "[bold green] Waiting for response...",
            spinner="dots12"
        ):
            response = Generation.call(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                model=model_called,
                messages=messages,
                result_format='message',
                stream = stream_output,
                incremental_output = stream_output
                # temporary = 0.7
            )
        if model_called == 'deepseek-r1':
            if stream_output == True:
                for chunk in response:
                    thinking = chunk.output.choices[0].message.reasoning_content
                    answering = chunk.output.choices[0].message.content
                    if(thinking !="" and answering == ""): 
                        if not reason_switch:
                            console.print("[bold blue]思考过程:[/]")
                            reason_switch = True
                        reasoning_content_tmp = chunk.output.choices[0].message.reasoning_content
                        reasoning_content_tmp_md = Markdown("".join(reasoning_content_tmp.splitlines()))
                        console.print(reasoning_content_tmp_md, end="")
                    elif(answering != ""):
                        if not answer_switch:
                            console.print("[bold blue]最终回答:[/]")
                            answer_switch = True
                        content_tmp = chunk.output.choices[0].message.content
                        content_tmp_md = Markdown(content_tmp)
                        console.print(content_tmp_md)
                        content += content_tmp
                log(f"AI:{content}")
                messages.append({'role': 'assistant', 'content': content})
            else:
                console.print("[bold blue]思考过程:[/]")
                reasoning_content = response.output.choices[0].message.reasoning_content
                reasoning_content_md = Markdown("".join(reasoning_content.splitlines()))
                console.print(reasoning_content_md, end="")

                console.print("[bold blue]最终回答:[/]")
                content = response.output.choices[0].message.content
                content_md = Markdown(content)
                console.print(content_md)
                log(f"AI:{content}")
                messages.append({'role': 'assistant', 'content': content})
        else:
            console.print("[bold blue]最终回答:[/]")
            content = response.output.choices[0].message.content
            content_md = Markdown(content)
            console.print(content_md)
            # Append model output to messages
            log(f"AI:{content}")
            messages.append({'role': 'assistant', 'content': content})
        round_count+=1

# Call coder model
def coder_model():
    """Call the coder model."""
    messages = [{'role': 'system', 'content': 'You are a programming master, Please output pure code without explanation and Markdown syntax'}]
    for _ in range(1):  # Append user input to messages
        in_content_goal = input("请输入目标：")
        in_content_require = input("请输入要求：")
        messages.append({'role': 'user', 'content': f'目标：{in_content_goal}'})
        messages.append({'role': 'user', 'content': f'要求：{in_content_require}'})
        response = Generation.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model=CODER_MODELS[0],
            messages=messages,
            result_format='message',
        )
        output = response.output.choices[0].message.content
        # Append model output to messages
        messages.append({'role': 'master', 'content': output})
        with open('exec.py', "w", encoding="utf-8") as file:
            file.write(output)
        print(output)

# Call visual model
def visual_model():
    """Call the visual model."""
    while True:
        image_path = input("图片地址：")
        in_content = input("请输入：")
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": image_path},
                    {"text": in_content}
                ]
            }
        ]
        response = dashscope.MultiModalConversation.call(
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            model=VISUAL_MODELS[0],
            messages=messages
        )
        print(response.output.choices[0].message.content)

def model_for_test():
    """Test model with user input."""
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
    while True:
        in_content_goal = input("请输入目标：")
        in_content_require = input("请输入要求：")
        # Append user input to messages
        messages.append({'role': 'user', 'content': f'目标：{in_content_goal}'})
        messages.append({'role': 'user', 'content': f'要求：{in_content_require}'})
        response = Generation.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model=CODER_MODELS[0],
            messages=messages,
            result_format='message',
        )
        output = response.output.choices[0].message.content
        # Append model output to messages
        messages.append({'role': 'assistant', 'content': output})
        print(output)

if __name__ == '__main__':
    general_model()
    # coder_model()
    # visual_model()