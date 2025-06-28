# 📟 Real-Time Anomaly Detection System with Streamlit + LangGraph + Telegram

This project demonstrates a real-time anomaly detection system for sensor data using a trained LSTM Autoencoder. It includes a **Streamlit dashboard** for visualization and a **LangGraph agent** for anomaly diagnosis and **Telegram notifications**.

---

## 🚀 Features

- Real-time simulation of sensor data streaming
- Anomaly detection with LSTM Autoencoder
- Agentic diagnostic pipeline with LangGraph
- Feature attribution for anomalies (top 3 contributors)
- Instant alerting via Telegram notifications
- Streamlit dashboard with live plotting and anomaly flags

---

## 🧠 Model Details

- **Model**: LSTM-based Autoencoder (trained with reconstruction loss)
- **Frameworks**: TensorFlow, LangGraph, Streamlit
- **Threshold**: 95th percentile of reconstruction error on normal data

---

## 📁 Project Structure

```
.
├── archive/MetroPT3(AirCompressor).csv   # Sensor dataset
├── agent.py                              # Agent logic and dashboard
├── app.py                                # Dashboard
├── autoencoder_model.keras               # Trained LSTM autoencoder model
├── .env                                  # Environment file for secrets
└── main.py                               # FastAPI server for inference
```

---

## 🔧 Setup

1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

2. **Add Environment Variables**

Create a `.env` file:

```env
TELEGRAM_BOT_TOKEN=<your_bot_token>
TELEGRAM_CHAT_ID=<your_chat_id>
```

3. **Run FastAPI Inference Server**

```bash
uvicorn main:app --reload
```

4. **Run the Streamlit App**

```bash
streamlit run langgraph_pipeline.py
```

---

## 📊 How It Works

- The Streamlit app streams data row-by-row from a CSV.
- Each sequence of 60 timesteps is sent to the FastAPI server.
- The LangGraph pipeline:
  - Runs inference using the LSTM autoencoder
  - Calculates reconstruction loss
  - Checks if it's above the 95th percentile threshold
  - Identifies top contributing features
  - Notifies via Telegram if anomaly detected

---

## 📬 Telegram Integration

- Set up a bot via `@BotFather`
- Get chat ID from `getUpdates`
- Used for customer-facing alerts when anomalies occur

---

## ✨ Example Notification

```
🚨 Anomaly Detected!
🕒 2025-06-28T21:15:33
📉 Loss: 0.008342

📌 Top Contributing Features:
- DV_pressure: 0.002309
- TP3: 0.001765
- Motor_current: 0.001102
```

---

## 🛠️ Future Enhancements

- Convert autoencoder to ONNX for faster inference
- Store flagged sequences in database
- Add support for real sensor ingestion (MQTT, Kafka, etc.)
- Enable auto-retraining on new data

---

## 📎 License

MIT License

---

Built with ❤️ by Ritika Banerjee for anomaly monitoring and intelligent diagnostics.
