# 🌦️ Weather AI Agent
An advanced conversational AI agent specializing in weather data, featuring three reasoning architectures (Basic, Chain-of-Thought, ReAct) with comparative evaluation. Integrates WeatherAPI and OpenAI's GPT models.

An intelligent agent that compares three AI reasoning approaches for weather queries:
1. **Basic** - Direct responses
2. **Chain-of-Thought** - Step-by-step reasoning
3. **ReAct** - Dynamic tool usage

## 🚀 Features

- **Multi-Architecture Comparison**  
  Evaluate Basic vs. CoT vs. ReAct responses side-by-side
- **Real Weather Data**  
  Powered by [WeatherAPI](https://www.weatherapi.com/)
- **Advanced Tool Usage**  
  - Current weather lookup  
  - Multi-day forecasts  
  - Mathematical calculations  
  - Simulated web search
- **Evaluation System**  
  Rate responses and save comparisons to CSV

## ⚡ Quick Start

### Prerequisites
- Python 3.8+
- [WeatherAPI Key](https://www.weatherapi.com/signup.aspx)
- OpenAI-compatible API key

### Installation
```bash
git clone https://github.com/yourusername/weather-ai-agent.git
cd weather-ai-agent
pip install -r requirements.txt
```
### Configuration
Create .env file:                     
API_KEY=your_openai_key                           
BASE_URL=https://api.openai.com/v1  # Or your custom endpoint                            
LLM_MODEL=gpt-4-mini               # Your preferred model                           
WEATHER_API_KEY=your_weather_key                                   

## 🖥️ Usage
Interactive Mode
```bash
python conversational_agent.py
```
### Menu Options: 
1. Single Agent (Basic/CoT/ReAct)
2. Comparative Evaluation

### Example Queries
- "What's the weather in Paris?"
- "Compare temperatures between Tokyo and London"
- "If it's 25°C in Rome and 18°C in Berlin, what's the difference in Fahrenheit?"

## 📊 Comparative Evaluation System
### Evaluation Flow
(Example evaluation workflow - replace with your actual screenshot)
-Runs all three agents simultaneously
-Displays responses in formatted table
-Collects user ratings
-Saves results to agent_evaluation.csv

## 🛠️ Technical Architecture

graph TD                      
    A[User Query] --> B{Agent Type}                            
    B --> C[Basic]                                     
    B --> D[Chain-of-Thought]                             
    B --> E[ReAct]                         
    C --> F[Direct Response]                       
    D --> G[Step-by-Step Reasoning]                                
    E --> H[Tool: WeatherAPI]                            
    E --> I[Tool: Calculator]                           
    E --> J[Tool: Web Search]                           

    
## 📂 File Structure

├── conversational_agent.py  # Main application                              
├── requirements.txt         # Dependencies                                   
├── .env.example             # Configuration template                            
├── agent_evaluation.csv     # Generated evaluations                     
└── README.md                           
