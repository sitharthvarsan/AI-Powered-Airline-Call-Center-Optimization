Problem 1 — Two-Agent Airline Information System

1. Installation:

  •	Navigate to the problem1/ directory.
  •	Install all required dependencies:
    pip install -r requirements.txt
  •	Add your OpenAI API key inside api_keys.env:
    OPENAI_API_KEY="your-api-key-here"
  •	Ensure the dataset indian_flights_dataset_2000_nozeros.csv is present inside the same folder.


2. Running the Code:

Run the main script directly:
python main.py

The script will automatically execute the required test cases:
  •	get_flight_info("AI123")
  •	info_agent_request("AI123")
  •	qa_agent_respond("When does Flight AI123 depart?")
  •	qa_agent_respond("What is the status of Flight AI999?")
Each function prints its strict JSON output or Python dictionary (for the first function).


3. Multi-Agent Function Calling (Approach Summary)

This project uses a simple two-agent design:
  •	Info Agent
  •	Implemented as an internal function (get_flight_info)
  •	Returns flight details strictly in JSON (via info_agent_request)
  •	QA Agent
  •	Parses the user’s natural-language question
  •	Extracts flight number using regex
  •	Calls the Info Agent
  •	Returns the final structured JSON response in the required format
This design simulates multi-agent coordination while keeping the system deterministic and easy to evaluate.