import chainlit as cl

@cl.on_chat_start
async def main():
    await cl.Message(content="Hello World").send()

@cl.on_message
async def on_message(message: cl.Message):
    response = f"Hello, you just sent: {message.content}!"
    await cl.Message(response).send()
