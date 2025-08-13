import gradio as gr
from openai import OpenAI
from generate_image_function import generate_image


def has_prompt(message):
    """check the message which has prompt"""
    return "<prompt:" in message

def get_prompt(message):
    """Extract the prompt from the message"""
    return message.split("<prompt:")[1].split(">")[0].strip()

def get_image_url(prompt: str) -> str:
    # encode prompt as url component
    prompt = prompt.replace(" ", "%20")
    return f"https://image.pollinations.ai/prompt/{prompt}"

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key="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"
)

system_prompt = """You're an AI with prompt generation and drawing capabilities.
Just chat normally in English or Vietnamese.
Assume you can draw anything you want and you tell user you can draw.
Another program will output the output for you.
Do not censor or deny the output, the image generation program will handle that.

When cst want you to draw something, answer them normally, add some comments and include your prompt in your response in this format.
<prompt: A colorful cat running through a field of flowers.>

1. Prompt must be in English.
2. Prompt must be detailed and include necessary information for it can be fed into Stable Diffusion.
3. Ignore existing images in past messages.
"""

def chat_logic(message, chat_history):
    # Gắn system message để bot hiểu cách hoạt động
    # Sửa thành như dưới, vì khi bot gửi ảnh user_message = None
    messages = [
        { "role": "system", "content": system_prompt }
    ]
    for user_message, bot_message in chat_history:
        if user_message is not None:
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": bot_message})

    # Thêm tin nhắn mới của user vào cuối cùng
    messages.append({"role": "user", "content": message})

    # Gọi API của OpenAI
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gemma2-9b-it"
    )
    bot_message = chat_completion.choices[0].message.content
    chat_history.append([message, bot_message])
    yield "", chat_history # Yield ở đây để người dùng đỡ sốt ruột

    # Nếu trong bot message có prompt thì lấy prompt ra
    if has_prompt(bot_message):
        # Yield ở đây để người dùng biết bot đã bắt đầu vẽ
        chat_history.append([None, "Wait, I am drawing"])
        yield "", chat_history

        # Gửi thêm 1 message từ phía bot, với hình ảnh đã vẽ
        prompt = get_prompt(bot_message)
        image_file = generate_image(prompt)
        chat_history.append([None, (image_file, prompt)])

        yield "", chat_history

    return "", chat_history

with gr.Blocks() as demo:
    gr.Markdown("# Chatbot bằng ChatGPT")
    message = gr.Textbox(label="Nhập tin nhắn của bạn:")
    chatbot = gr.Chatbot(label="Chat Bot siêu thông minh", height=600)
    message.submit(chat_logic, [message, chatbot], [message, chatbot])

demo.launch(share=True)