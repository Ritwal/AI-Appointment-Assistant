# AI-Appointment-Assistant

An AI-powered appointment assistant built using LangGraph, LangChain, OpenAI GPT-4o, and Streamlit. It interacts with users, detects intent, extracts appointment details, and guides them through the booking process.



# Workflow 

![image](https://github.com/user-attachments/assets/33611cff-9ed1-4887-bf32-435e18f62c5a)

---


## How to Use This Assistant

### 1. Clone and Setup
```bash
git clone https://github.com/Ritwal/AI-Appointment-Assistant.git
cd your-repo-folder
create python environment 
pip install -r requirements.txt
open workflow.py
paste your openai API key in line 11 and save it
streamlit run Assistant_app.py

-----
# Chat sample

**User** - hi
**AI** - Hello! How can I assist you today?
**User** - book appointment
**AI** - Sure, what date and time would you like to book your appointment?
**User** - 30 june
**AI** - Hello! It seems you're looking to book an appointment for June 30th. Could you please provide more details, such as the type of appointment and any specific time you have in mind?
**User** - 9 am, vistual
**AI** - ✅ Your Virtual appointment is confirmed for 2025-06-30 at 9:00 AM. You'll receive a calendar invite shortly.
**AI** - Is there anything else I can help you with?
**User** - what service do you provide
**AI** - Hello! I'm here to assist with a wide range of inquiries, such as providing information, answering questions, helping with scheduling, and offering guidance on various topics. If you have a specific question or need assistance with something, feel free to let me know!



