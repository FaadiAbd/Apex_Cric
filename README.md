🏏 BowlForm AI – Cricket Bowling Action Analyzer

An AI-powered web application that analyzes cricket bowling actions using computer vision and provides biomechanical feedback with Large Language Model insights.

🚀 Overview

BowlForm AI allows users to upload a side-view bowling video and receive:

🎥 Pose-based skeleton visualization

📊 Biomechanical metrics at release point

🤖 AI-generated coaching feedback

📁 Persistent analysis history

👤 Secure user authentication

The system combines MediaPipe Pose Detection with LLM-based coaching feedback to simulate a virtual cricket biomechanics coach.

🧠 How It Works

User uploads bowling video.

MediaPipe Pose Landmarker detects 3D body landmarks.

Release frame is identified using elbow extension logic.

Biomechanical features are extracted:

Elbow Angle

Shoulder Angle

Arm Verticality

Front Knee Angle

Stride Length

Features are sent to DeepSeek (via OpenRouter API).

AI returns short actionable coaching tips.

Results and processed videos are saved for history tracking.

🏗 System Architecture

Frontend:

HTML (Jinja2 Templates)

Tailwind-style UI

JavaScript for dynamic updates

Backend:

Flask (Python)

SQLite Database

MediaPipe Pose Landmarker (Tasks API)

OpenRouter API (DeepSeek Model)

Video Processing:

OpenCV

Frame-by-frame landmark extraction

Skeleton overlay rendering

Release frame metric detection

📂 Project Structure
cricket-bowling-analyzer/
│
├── Complete_Project.py
├── templates/
│   ├── login.html
│   ├── dashboard.html
│   ├── analysis_page.html
│   ├── profile.html
│   └── sections/
│
├── static/
│   ├── js/
│   ├── css/
│   ├── profile_photos/
│   └── output videos
│
├── uploads/
├── models/
├── requirements.txt
└── .env
🔐 Features
👤 Authentication System

User Signup

Secure Login

Profile Management

Persistent Session Tokens

🎥 Video Analysis

Upload MP4/MOV/AVI

Automatic pose detection

Skeleton-only video output

Full overlay video output

📊 Biomechanics Metrics

Elbow extension angle

Arm verticality

Front knee stability

Stride normalization

Shoulder alignment

🤖 AI Coaching Feedback

Generated via DeepSeek LLM

Actionable, concise tips

Focus on:

Balance

Power

Injury prevention

Efficiency

📁 History Tracking

All analyses saved

View previous sessions

Persistent storage in SQLite

🛠 Installation Guide
1️⃣ Clone Repository
git clone https://github.com/yourusername/bowlform-ai.git
cd bowlform-ai
2️⃣ Create Virtual Environment
python -m venv venv

Activate:

Windows:

.\venv\Scripts\activate

Mac/Linux:

source venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Create .env File

Create a file named .env in the root directory:

OPENROUTER_API_KEY=your_openrouter_key_here
MODEL_ID=deepseek/deepseek-chat
SECRET_KEY=your_secret_key
5️⃣ Run the Application
python Complete_Project.py

Open in browser:

http://127.0.0.1:5001
🧪 Technologies Used

Python

Flask

OpenCV

MediaPipe Tasks API

SQLite

NumPy

DeepSeek LLM

OpenRouter API

HTML / CSS / JavaScript

📈 Future Improvements

Real-time webcam analysis

Multi-angle pose fusion

Speed detection integration

Injury risk prediction model

Player comparison analytics

Mobile deployment

🎓 Academic Value

This project demonstrates:

Applied Computer Vision

Biomechanical Feature Engineering

LLM Integration in Sports Analysis

Full-stack Web Development

Secure User Authentication

Video Processing Pipeline Design

⚠ Disclaimer

This tool provides AI-generated feedback and should not replace professional coaching. Results depend on video quality and camera angle.
Steps to Run the ApexCric Project (Windows)
1. Extract the Project

Download and extract the project ZIP file.
Open the extracted folder in VS Code or Command Prompt.

Your project folder structure will look like this:

APEX_CRIC-SUB/
│
├── models/
├── static/
├── templates/
├── uploads/
├── venv/
│
├── .env
├── .gitignore
├── apex_cric.db
├── Complete front End.html
├── Complete_Backend.py
├── Complete_Project.py
├── LICENSE
├── README.md
├── requirements.txt

Open Command Prompt and navigate to the project folder:

cd APEX_CRIC-SUB
2. Create a Virtual Environment (If not already created)

If the venv folder already exists, you can skip this step.

Otherwise create a virtual environment:

python -m venv venv
3. Activate the Virtual Environment

Activate the environment using:

venv\Scripts\activate

After activation, the terminal should show:

(venv)

before the command prompt.

4. Install Project Dependencies

Install all required libraries using the requirements file:

pip install -r requirements.txt

This installs libraries such as:

Flask
OpenCV
NumPy
MediaPipe
Requests
Python-dotenv
ReportLab
QRCode
Pillow
5. Configure Environment Variables

Ensure the .env file contains the required API keys:

OPENROUTER_API_KEY=your_openrouter_api_key
MODEL_ID=deepseek/deepseek-chat
SECRET_KEY=your_secret_key

These variables are required for the AI feedback and training plan generation.

6. Database Setup

The project uses SQLite.

The database file:

apex_cric.db

is automatically created and managed by the application.
No manual database configuration is required.

7. Run the Application

Start the backend server by running:

python Complete_Project.py

The terminal should display something similar to:

Running on http://127.0.0.1:5001
8. Open the Application

Open your browser and go to:

http://localhost:5001

You will see the ApexCric login page.

9. First-Time Model Download

When the analysis feature is used for the first time, the system automatically downloads the MediaPipe Pose Landmarker model and stores it in the models folder.

This process happens automatically and requires an internet connection.

10. Using the System
Create a user account.
Log in to the dashboard.
Upload a bowling video.
The system processes the video using pose detection.
AI generates feedback, training plans, and tactical advice.
Results can be viewed, downloaded, or shared.
