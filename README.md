# LLM Chatbot with Gemini  & Flask

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Live Demo](https://img.shields.io/badge/demo-live-green)](https://llm-chatbot-9.onrender.com)

A real-time chatbot leveraging Google's Gemini for document analysis of PDFs, built with Flask and WebSockets.

![Chat Interface](https://llm-chatbot-9.onrender.com/)

## Features

- Real-time WebSocket communication
- Multi-PDF analysis (Mobily AR & Caterpillar manuals)
- Conversation memory with session management
- Error handling with user-friendly messages
- Deployment-ready configuration

## Prerequisites

- Python 3.10+
- Git
- [Google Cloud Account](https://cloud.google.com/)
- [Render Account](https://render.com/) (for deployment)

## Installation

```bash
git clone https://github.com/your-username/llm-chatbot.git
cd llm-chatbot
pip install -r requirements.txt
