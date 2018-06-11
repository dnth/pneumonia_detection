#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Simple Bot to reply to Telegram messages.
This program is dedicated to the public domain under the CC0 license.
This Bot uses the Updater class to handle the bot.
First, a few handler functions are defined. Then, those functions are passed to
the Dispatcher and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.
Usage:
Basic Echobot example, repeats messages.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import logging

from bot import load_pneumonia_model, predict_pneumonia
model = load_pneumonia_model('pneumonia_finetunevgg.h5')


# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments bot and
# update. Error handlers also receive the raised TelegramError object in error.
def start(bot, update):
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi! I am a bot designed to diagnose pneumonia. Send a me a picture of a xray film and I will tell you what i think!')


def help(bot, update):
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def echo(bot, update):
    """Echo the user message."""
    update.message.reply_text(update.message.text)

def check_message(bot, update):
    update.message.reply_text("Let me check your message")


def diagnose(bot, update):
    update.message.reply_text("I have got the picture! Analysing..")
    file_id = update.message.photo[-1].file_id
    newFile = bot.getFile(file_id)
    picture = newFile.download('test.jpg')
    probability_of_pneumonia = predict_pneumonia(picture)

    percent_pneumonia = probability_of_pneumonia * 100.0

    update.message.reply_text("Probability of pneumonia: %2f %%" % percent_pneumonia)


def error(bot, update, error):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, error)


def main():
    """Start the bot."""
    # Create the EventHandler and pass it your bot's token.
    updater = Updater("598287772:AAGFgwsgB5RmZT0eyYB5ehufm_G-YA7D5zo")

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.text, start))

    dp.add_handler(MessageHandler(Filters.forwarded & Filters.photo, diagnose))

    dp.add_handler(MessageHandler(Filters.photo & (~ Filters.forwarded), diagnose))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()