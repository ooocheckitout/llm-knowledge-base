import logging.handlers
import os
import sys
from pathlib import Path
import re

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from src.common.services.chat import ChatService

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

chat_service = ChatService()


async def echo(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    print(update.effective_user)
    print(update.effective_message)
    print(update.effective_chat)
    print(update.effective_sender)

    # AI-ish (is_topic_message=True, message_thread_id=223)
    # Message(channel_chat_created=False, chat=Chat(id=-1002451479256, is_forum=True, title='Яхта, море дискотека', type=<ChatType.SUPERGROUP>), date=datetime.datetime(2025, 7, 12, 13, 36, 2, tzinfo=datetime.timezone.utc), delete_chat_photo=False, from_user=User(first_name='Oleh', id=213260575, is_bot=False, language_code='en', username='hearthfire'), group_chat_created=False, is_topic_message=True, message_id=272, message_thread_id=223, reply_to_message=Message(channel_chat_created=False, chat=Chat(id=-1002451479256, is_forum=True, title='Яхта, море дискотека', type=<ChatType.SUPERGROUP>), date=datetime.datetime(2025, 7, 12, 12, 20, 16, tzinfo=datetime.timezone.utc), delete_chat_photo=False, forum_topic_created=ForumTopicCreated(icon_color=7322096, name='AI-ish'), from_user=User(first_name='Oleh', id=213260575, is_bot=False, language_code='en', username='hearthfire'), group_chat_created=False, is_topic_message=True, message_id=223, message_thread_id=223, supergroup_chat_created=False), supergroup_chat_created=False, text='hi')

    # General
    # Message(channel_chat_created=False, chat=Chat(id=-1002451479256, is_forum=True, title='Яхта, море дискотека', type=<ChatType.SUPERGROUP>), date=datetime.datetime(2025, 7, 12, 13, 36, 17, tzinfo=datetime.timezone.utc), delete_chat_photo=False, from_user=User(first_name='Oleh', id=213260575, is_bot=False, language_code='en', username='hearthfire'), group_chat_created=False, message_id=273, supergroup_chat_created=False, text='noice')

    # наше желище (is_topic_message=True, message_thread_id=2)
    # Message(channel_chat_created=False, chat=Chat(id=-1002451479256, is_forum=True, title='Яхта, море дискотека', type=<ChatType.SUPERGROUP>), date=datetime.datetime(2025, 7, 12, 13, 37, 49, tzinfo=datetime.timezone.utc), delete_chat_photo=False, from_user=User(first_name='Oleh', id=213260575, is_bot=False, language_code='en', username='hearthfire'), group_chat_created=False, is_topic_message=True, message_id=274, message_thread_id=2, reply_to_message=Message(channel_chat_created=False, chat=Chat(id=-1002451479256, is_forum=True, title='Яхта, море дискотека', type=<ChatType.SUPERGROUP>), date=datetime.datetime(2025, 2, 15, 19, 29, 39, tzinfo=datetime.timezone.utc), delete_chat_photo=False, forum_topic_created=ForumTopicCreated(icon_color=7322096, name='наше желище'), from_user=User(first_name='Liliia', id=537022616, is_bot=False, last_name='Leshchynska', username='lleshchynska'), group_chat_created=False, is_topic_message=True, message_id=2, message_thread_id=2, supergroup_chat_created=False), supergroup_chat_created=False, text='tss')


async def answer_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("answer_reply {} ({})".format(update.effective_user.full_name, update.effective_user.id))

    local_file_path = Path(".documents") / str(
        update.effective_chat.id) / update.effective_message.reply_to_message.document.file_name
    local_file_path.parent.mkdir(parents=True, exist_ok=True)

    file = await context.bot.get_file(update.effective_message.reply_to_message.document)
    await file.download_to_drive(local_file_path)

    context = f"Document {local_file_path.name}\n\n"
    documents = await PyPDFLoader(local_file_path).aload()
    for index, document in enumerate(documents, start=1):
        context += f"Content of Page {index}:\n\"{document.page_content}\"\n\n"

    with open(local_file_path.with_suffix(f".context.txt"), 'w', encoding='utf-8') as f:
        f.write(context)

    #     prompt = f"""
    # You (@lileg_ai_bot) are a helpful assistant.
    # Structure the information in document in paragraphs - general document description, page by page details, potential questions about the document to keep the conversation going.
    # Don't include the user question in the answer.
    # Avoid custom formatting and make the answer more human to look more like a conversation.
    # Make it concise up to 5 sentences and specific to the user question.
    # Use the following context to answer the question: {context}\n\nQuestion: {update.effective_message.text}
    # """
    prompt = """
You (@lileg_ai_bot) are a personal AI assistant for the user. You have access to the following information to help answer their question:

- Document Context - the relevant text extracted from user’s PDFs or documents here.
<start>
{{ document_context }}
<end>

- Conversation History - a brief summary of recent chat history or relevant previous messages.
<start>
{{ conversation_history }}
<end>

- Additional Context - any other relevant context like user preferences or profile info
<start>
{{ additional_context }}
<end>

Follow all these guidelines when answering:
1. Use the context: If the documents or chat history contain the answer, use that information directly. Quote or paraphrase from it as needed to answer the question.
2. Fallback to general knowledge: If the context doesn’t have the answer, rely on your own knowledge to help, but still keep the response brief and relevant.
3. Be concise: Provide a short answer using as few words as necessary to fully address the question. Do not add unnecessary detail or length, especially if the answer is clearly in the context.
4. Simple language: Use clear, simple words. Avoid jargon or complicated terms when a simple phrase will do. The tone should be friendly and human-like, not overly formal.
5. Personal tone: Address the user in a warm and personal manner (e.g. “Hi there, ...”). Make the user feel you are their personal assistant who knows their context. Use first-person ("I") for yourself and second-person ("you") for the user.
6. No extra formatting: Do not include any markdown or special formatting characters (like `*`, `_`, or HTML tags) in your reply, unless the user specifically asks for formatted output. Just write plain text in a normal conversational style.
7. No chain-of-thought: Do not reveal any internal reasoning or `<think>` sections. Provide only the final answer to the user’s question, starting directly with the answer. (Absolutely no `"Step 1:"` or analysis before the answer.)
8. Follow-up question: After answering, always ask a brief, relevant question to the user to keep the conversation going.

Now, using the above guidelines, answer the user's question.

{{ user_question }}
"""
    final_prompt = (
        prompt
        .replace("{{ document_context }}", context)
        .replace("{{ conversation_history }}", "")
        .replace("{{ additional_context }}", "")
        .replace("{{ user_question }}", update.effective_message.text)
    )

    with open(local_file_path.with_suffix(f".prompt.txt"), "w", encoding='utf-8') as f:
        f.write(prompt)

    completion = chat_service.ask(final_prompt)

    with open(local_file_path.with_suffix(f".completion.txt"), "w", encoding='utf-8') as f:
        f.write(completion.content)

    without_think_block = re.sub(r"<think>.*?</think>\n?", "", completion.content, flags=re.DOTALL)

    await update.effective_message.reply_markdown(
        without_think_block,
        reply_to_message_id=update.effective_message.message_id
    )


async def answer(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("answer {} ({})".format(update.effective_user.full_name, update.effective_user.id))

    completion = chat_service.ask(update.effective_message.text)

    await update.effective_message.reply_text(
        completion.content,
        reply_to_message_id=update.effective_message.message_id
    )


app = ApplicationBuilder().token(os.getenv('TELEGRAM_BOT_TOKEN')).build()

# app.add_handler(MessageHandler(filters.ALL, echo))
# app.add_handler(MessageHandler(filters.ALL, echo))

# app.add_handler(
#     MessageHandler(
#         filters.Mention(os.getenv('TELEGRAM_BOT_USERNAME')) & filters.Document.PDF, remember
#     )
# )

app.add_handler(
    MessageHandler(
        filters.Mention(os.getenv('TELEGRAM_BOT_USERNAME')) & filters.REPLY, answer_reply
    )
)

app.add_handler(
    MessageHandler(
        filters.Mention(os.getenv('TELEGRAM_BOT_USERNAME')), answer
    )
)

app.run_polling()
