#!/usr/bin/env python3
"""
Manager-bot: Telegram AI Assistant powered by Google Gemini.
Built with python-telegram-bot (v20+) and google-generativeai.
"""

import os
import sys
import logging
from typing import List

from dotenv import load_dotenv
import google.generativeai as genai
from telegram import Update
from telegram.constants import ChatAction, ChatType
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("ManagerBot")

# ---------------------------------------------------------------------------
# Environment & Configuration Setup
# ---------------------------------------------------------------------------
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
AI_API_KEY = os.getenv("AI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

if not TELEGRAM_BOT_TOKEN:
    logger.critical("TELEGRAM_BOT_TOKEN is missing in environment variables.")
    sys.exit(1)

if not AI_API_KEY:
    logger.critical("AI_API_KEY is missing in environment variables.")
    sys.exit(1)

# Configure Google Gemini
genai.configure(api_key=AI_API_KEY)

# System instruction defining the personality and scope of Manager-bot
SYSTEM_INSTRUCTION = (
    "You are Manager-bot, an intelligent, professional, and helpful Telegram AI assistant. "
    "Provide clear, accurate, concise, and context-aware responses. Format your replies "
    "using clean formatting suitable for chat messages."
)

gemini_model = genai.GenerativeModel(
    model_name=GEMINI_MODEL_NAME,
    system_instruction=SYSTEM_INSTRUCTION,
)

# Telegram message length limit is 4096 characters
TELEGRAM_MAX_MESSAGE_LENGTH = 4000


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def split_message(text: str, max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> List[str]:
    """Splits a long message into safe Telegram-sized chunks."""
    if len(text) <= max_length:
        return [text]

    chunks = []
    while len(text) > max_length:
        # Try to split at the last newline before the limit
        split_idx = text.rfind("\n", 0, max_length)
        if split_idx == -1:
            # Fallback: split at the last space
            split_idx = text.rfind(" ", 0, max_length)
        if split_idx == -1:
            # Forced cut if no space or newline is found
            split_idx = max_length

        chunks.append(text[:split_idx].strip())
        text = text[split_idx:].strip()

    if text:
        chunks.append(text)
    return chunks


# ---------------------------------------------------------------------------
# Telegram Handlers
# ---------------------------------------------------------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /start command."""
    user = update.effective_user
    welcome_text = (
        f"👋 Hello, {user.first_name if user else 'there'}!\n\n"
        "I am **Manager-bot**, your intelligent AI assistant powered by Google Gemini.\n\n"
        "✨ **How to use me:**\n"
        "• **Private Chat:** Send me any question or prompt directly.\n"
        "• **Groups:** Mention me (@bot_username) or reply to any of my messages.\n"
        "• **Channels:** Add me as an Administrator with post rights.\n\n"
        "Type /help to learn more."
    )
    if update.effective_message:
        await update.effective_message.reply_text(welcome_text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /help command."""
    help_text = (
        "🛠 **Manager-bot Assistance**\n\n"
        "1. Ask general knowledge, coding, drafting, or problem-solving queries.\n"
        "2. To talk in groups: Tag me or reply directly to my messages.\n"
        "3. Need a fresh topic? Just ask your new question directly!"
    )
    if update.effective_message:
        await update.effective_message.reply_text(help_text)


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Processes incoming text messages from Private chats, Groups, and Channels."""
    message = update.effective_message
    if not message or not message.text:
        return

    chat = update.effective_chat
    user_query = message.text.strip()
    bot_username = context.bot.username

    # Group / Supergroup logic:
    # Only reply if the bot is tagged or directly replied to (avoids unwanted group spam)
    if chat.type in [ChatType.GROUP, ChatType.SUPERGROUP]:
        is_replied_to_bot = (
            message.reply_to_message
            and message.reply_to_message.from_user
            and message.reply_to_message.from_user.id == context.bot.id
        )
        is_mentioned = bot_username and f"@{bot_username}" in user_query

        if not (is_replied_to_bot or is_mentioned):
            return  # Ignore messages not directed at the bot

        # Clean mention tag from prompt
        if is_mentioned:
            user_query = user_query.replace(f"@{bot_username}", "").strip()

    if not user_query:
        return

    # Indicate activity
    await chat.send_action(action=ChatAction.TYPING)

    try:
        # Asynchronously call the Gemini API
        response = await gemini_model.generate_content_async(user_query)

        if not response.text:
            await message.reply_text("I could not generate an answer for that query.")
            return

        # Split into safe chunks and send
        chunks = split_message(response.text)
        for chunk in chunks:
            await message.reply_text(chunk)

    except Exception as exc:
        logger.error("Error communicating with Gemini API: %s", exc, exc_info=True)
        await message.reply_text(
            "⚠️ An error occurred while processing your request. Please try again later."
        )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global error handler for unhandled exceptions."""
    logger.error("Exception while handling an update: %s", context.error, exc_info=context.error)


# ---------------------------------------------------------------------------
# Application Initialization & Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for Manager-bot."""
    logger.info("Initializing Manager-bot...")

    # Build the PTB Application
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Register Command Handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))

    # Register Message Handler (covers Private, Groups, and Channel posts)
    app.add_handler(
        MessageHandler(
            (filters.TEXT & (~filters.COMMAND)) | filters.UpdateType.CHANNEL_POST,
            handle_text_message,
        )
    )

    # Register Global Error Handler
    app.add_error_handler(error_handler)

    logger.info("Manager-bot is live and listening for messages.")
    # Run long polling
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
