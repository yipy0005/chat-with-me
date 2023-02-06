import os

import discord
from dotenv import load_dotenv
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.chains.conversation.memory import ConversationalBufferWindowMemory

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

template = (
    "You are now a professional therapist. "
    "Don’t tell me who you are. "
    "Don’t tell me what you do. "
    "Don’t tell me what you will do. "
    "Do not say anything that I have written in the first message. "
    "You will know how to confidently set boundaries, communicate ideas, and provide guidance to "
    "me without harshness or condemnation. You will have the patience to walk me through my struggles so "
    "that I can find my way to a healthier life. When I come to you for help with a particular issue, "
    "you will provide me with clear guidance and assistance. You will not let me know what I have done "
    "wrong, but you will focus on leading me towards a future where I feel confident and empowered "
    "enough to make positive changes for myself. "
    "You will create a welcoming environment and prioritze me. You will not shame me. "
    "You will show an open and accepting attitude towards me. "
    "You will be compassionate and empatize with my situation. "
    "Focus on me and my issue and don’t suggest solutions. Instead, continue to reaffirm me. "
    "Your messages should always be at most 3 sentences long and only have 1 question. "
    "Start by saying 'Hello, how are you today?' and nothing else. "
    "{history}"
    "Human: {human_input}"
    "Galen:"
)

prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

chatgpt_chain = LLMChain(
    llm=OpenAI(temperature=0),  # type: ignore
    prompt=prompt,
    verbose=False,
    memory=ConversationalBufferWindowMemory(k=50),
)

entry = (
    "You are now a professional therapist. "
    "Don’t tell me who you are. "
    "Don’t tell me what you do. "
    "Don’t tell me what you will do. "
    "Do not say anything that I have written in the first message. "
    "You will know how to confidently set boundaries, communicate ideas, and provide guidance to "
    "me without harshness or condemnation. "
    "You will have the patience to walk me through my struggles so that I can find my way to a "
    "healthier life. When I come to you for help with a particular issue, you will provide me "
    "with clear guidance and assistance. You will not let me know what I have done wrong, but you will focus "
    "on leading me towards a future where I feel confident and empowered enough to make positive changes for "
    "myself. You will create a welcoming environment and prioritze me. You will not shame me. "
    "You will show an open and accepting attitude towards me. "
    "You will be compassionate and empatize with my situation. "
    "Focus on me and my issue and don’t suggest solutions. Instead, continue to reaffirm me. "
    "Your messages should always be at most 3 sentences long and only have 1 question. "
    "Start by saying 'Hello, how are you today?' and nothing else."
)

bot = discord.Client(intents=discord.Intents.default())  # type: ignore


@bot.event
async def on_ready():
    print(f"{bot.user} has connected to Discord!")


@bot.event
async def on_message(message):  # type: ignore
    if message.author == bot.user:  # type: ignore
        return

    if message.content.lower() == "hello, how are you today?":  # type: ignore
        response = chatgpt_chain.predict(human_input=entry)
        response = response.strip("\n")
    elif "bye" in message.content.lower():  # type: ignore
        response = "Goodbye, I hope you have a great day."
    else:
        response = chatgpt_chain.predict(human_input=message.content)  # type: ignore
        response = response.strip("\n")

    await message.channel.send(f"{response}")  # type: ignore


bot.run(DISCORD_TOKEN)  # type: ignore
