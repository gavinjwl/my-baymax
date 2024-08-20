import chainlit as cl
import torch
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from chainlit.types import ThreadDict

torch.random.manual_seed(0)


@cl.on_chat_start
async def main():
    '''
    The on_chat_start decorator is used to define a hook that is called when a new chat session is created.
    '''

    model_id = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",
    )
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=100,
        top_k=50,
        temperature=0.1
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # llm = HuggingFacePipeline.from_model_id(
    #     model_id=model_id,
    #     token=False,
    #     task="text-generation",
    #     device_map="auto",
    #     # torch_dtype="auto",
    #     # trust_remote_code=True,
    #     pipeline_kwargs={
    #         "max_new_tokens": 100,
    #         "top_k": 50,
    #         "temperature": 0.1,
    #         "do_sample": False,
    #     },
    # )

    chat_model = ChatHuggingFace(llm=llm)
    cl.user_session.set("chat_model", chat_model)


# @cl.on_chat_resume
# async def on_chat_resume(thread: ThreadDict):
#     '''
#     The on_chat_resume decorator is used to define a hook that is called when a user resumes a chat session that was previously disconnected.
#     This can only happen if authentication and data persistence are enabled.
#     '''
#     print("The user resumed a previous chat session!")


@cl.on_message
async def on_message(message: cl.Message):
    '''
    The on_message decorator is used to define a hook that is called when a new message is received from the user.
    '''
    chat_model = cl.user_session.get("chat_model")

    # generation_args = {
    #     "max_new_tokens": 500,
    #     "return_full_text": False,
    #     "temperature": 0.0,
    #     "do_sample": False,
    # }
    # output = pipe(message.content, **generation_args)
    # generated_text = output[0]['generated_text']
    # print(generated_text)

    resp = chat_model.invoke(message.content)
    print(resp)
    # response = f"Hello, you just sent: {message.content}!"
    await cl.Message(resp.content).send()


@cl.on_stop
def on_stop():
    '''
    The on_stop decorator is used to define a hook that is called when the user clicks the stop button while a task was running.
    '''
    print("The user wants to stop the task!")


@cl.on_chat_end
def on_chat_end():
    '''
    The on_chat_resume decorator is used to define a hook that is called when a user resumes a chat session that was previously disconnected.
    This can only happen if authentication and data persistence are enabled.
    '''
    print("The user disconnected!")
