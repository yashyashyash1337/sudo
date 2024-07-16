import os
import random
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputTextMessageContent
from telegram.ext import Application, CommandHandler, ContextTypes, ConversationHandler, CallbackQueryHandler, MessageHandler, filters
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Define states for conversation
ASK_VALUES = range(1)

# Initialize Telegram bot
TOKEN = '7222048260:AAFs0BmdQyoX7Kmx1ANqaJH5LWAGTvc1esI'
YOUR_ADMIN_USER_ID = 'YOUR_ADMIN_USER_ID'

# Function to start the conversation
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text('Welcome! Enter the past 20-30 games crash/slides values separated by spaces:')
    return ASK_VALUES

# Function to handle user input of game crash values
async def ask_values(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        values = list(map(float, update.message.text.split()))
        if len(values) < 20:
            raise ValueError
    except ValueError:
        await update.message.reply_text("Invalid input. Please enter at least 20 numerical values separated by spaces.")
        return ASK_VALUES

    await update.message.reply_text("Processing...")

    try:
        prediction, risk_prediction, safe_prediction = predict_crash(values)
        response_text = (f"Prediction Results:\n"
                         f"🔮 Predicted Value: {prediction:.2f}\n"
                         f"⚠️ High Risk Prediction: {risk_prediction:.2f}\n"
                         f"✅ Safe Prediction: {safe_prediction:.2f}")
        await update.message.reply_text(response_text)
    except Exception as e:
        await update.message.reply_text(f"An error occurred: {str(e)}")

    return ConversationHandler.END

def predict_crash(crash_values):
    data = pd.DataFrame(crash_values, columns=['value'])
    data['time'] = np.arange(len(data))

    # Feature engineering
    data['rolling_mean'] = data['value'].rolling(window=5).mean().fillna(method='bfill')
    data['rolling_std'] = data['value'].rolling(window=5).std().fillna(method='bfill')
    data['diff'] = data['value'].diff().fillna(0)

    # Prepare training and testing data
    X = data[['time', 'rolling_mean', 'rolling_std', 'diff']]
    y = data['value']

    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    # Train XGBoost model
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_xgb_model = grid_search.best_estimator_

    # Train LSTM model
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')

    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)

    # Make predictions
    next_time = len(data)
    last_row = data.iloc[-1]
    next_features = np.array([[next_time, last_row['rolling_mean'], last_row['rolling_std'], last_row['diff']]])
    next_features_scaled = scaler.transform(next_features)
    next_features_lstm = next_features_scaled.reshape((next_features_scaled.shape[0], next_features_scaled.shape[1], 1))

    xgb_prediction = best_xgb_model.predict(next_features_scaled)[0]
    lstm_prediction = lstm_model.predict(next_features_lstm)[0][0]

    # Combine predictions
    combined_prediction = (xgb_prediction + lstm_prediction) / 2

    # Risk Prediction (High Risk)
    risk_prediction = combined_prediction * 1.5

    # Safe Prediction (Low Risk)
    safe_prediction = combined_prediction * 0.75

    return combined_prediction, risk_prediction, safe_prediction

async def welcome_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await update.message.reply_text(f'Hello {user.first_name}!')

async def broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user.id != YOUR_ADMIN_USER_ID:
        await update.message.reply_text('Unauthorized')
        return

    try:
        message = update.message.text.split(' ', 1)[1]
    except IndexError:
        await update.message.reply_text('Usage: /broadcast <message>')
        return

    try:
        chat_members = await context.bot.get_chat_members_count(update.effective_chat.id)
        for user_id in range(chat_members):
            try:
                await context.bot.send_message(chat_id=user_id, text=message)
            except Exception as e:
                print(f"Error sending message to {user_id}: {e}")

        await update.message.reply_text('Broadcast message sent to all group members.')
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

def main() -> None:
    application = Application.builder().token(TOKEN).read_timeout(30).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            ASK_VALUES: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_values)],
        },
        fallbacks=[CommandHandler('start', start)],
        per_chat=True
    )

    application.add_handler(conv_handler)
    application.add_handler(MessageHandler(filters.ChatType.PRIVATE, welcome_message))
    application.add_handler(CommandHandler('broadcast', broadcast))

    application.run_polling()

if __name__ == '__main__':
    main()
