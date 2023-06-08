import logging
import os
from aiogram import Bot, Dispatcher, types
from aiogram.types import ParseMode
from aiogram.utils import executor
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
from translate import Translator


# Configure logging
logging.basicConfig(level=logging.INFO)

# Get the OpenAI API key
os.environ["OPENAI_API_KEY"] = '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'
openai_api_key = os.environ["OPENAI_API_KEY"]
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize the bot and dispatcher
bot = Bot(token="6064761154:AAHpetSzNO5SEqNGfrZbph7vBErsWPn9TFQ")
dp = Dispatcher(bot)

# Define the AI model
max_input_size = 4096
num_outputs = 512
max_chunk_overlap = 20
chunk_size_limit = 600
prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

# Load the index
directory_path = "docs"
documents = SimpleDirectoryReader(directory_path).load_data()
index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

# Define the function for translating text
def translate_text(text, lang_from='en', lang_to='ru'):
    translator= Translator(to_lang=lang_to)
    translation = translator.translate(text)
    return translation

# Define the function for translating model response to Russian
def translate_response(response):
    translated_response = translate_text(response, lang_from='en', lang_to='ru')
    return translated_response

# Define the start command handler
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("Привет! Я чат-бот, который может с тобой общаться на русском языке. Просто отправь мне сообщение и я постараюсь ответить как можно лучше.")

# Define the message handler
@dp.message_handler()
async def echo_message(message: types.Message):
    await bot.send_message(chat_id=message.chat.id, text = '⌛️')

    input_text = message.text

    # Translate input text to English
    input_text_en = translate_text(input_text)

    # Query the index with the input text
    response_en = index.query(input_text_en, response_mode="compact").response

    # Translate response back to Russian
    response = translate_response(response_en)

    # Send the response back to the user
    await message.reply(response, parse_mode=ParseMode.HTML)

# Start the bot
if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
