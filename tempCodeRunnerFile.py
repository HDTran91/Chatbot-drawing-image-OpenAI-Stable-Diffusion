import gradio as gr
from openai import OpenAI

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key="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"
)

def chat_logic(message, chat_history):
    messages = []
    for user_message, bot_message in chat_history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": bot_message})

    messages.append({"role": "user", "content": message})
    chat_completetion = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=messages,
        stream=True
    )

    chat_history.append([message, "loading..."])
    yield "", chat_history  # Clear input, return updated history
    chat_history[-1][1] = ""

    for chunk in chat_completetion:
        delta = chunk.choices[0].delta.content or ""
        chat_history[-1][1] += delta
        yield "", chat_history

    return "", chat_history  # Final return to clear input and update chat
with gr.Blocks() as demo:
    gr.Markdown("# Chatbox AI")

    chatbot = gr.Chatbot(label="Smart Chatbox AI")
    message = gr.Textbox(label="Type your message here", placeholder="Enter your message...")
    state = gr.State([])  # Stores the chat history

    message.submit(
        fn=chat_logic,
        inputs=[message, state],
        outputs=[message, chatbot]  # Clear message box, update chat
    )

demo.launch(share=True)