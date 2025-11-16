import os
import json
import re
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables & initialize OpenAI client

load_dotenv("api_keys.env")
API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=API_KEY
)

# Load and prepare flight dataset

df = pd.read_csv("indian_flights_dataset_2000_nozeros.csv")

flight_database = {
    row["flight_number"]: {
        "flight_number": row["flight_number"],
        "departure_time": row["departure_time"],
        "destination": row["destination"],
        "status": row["status"]
    }
    for _, row in df.iterrows()
}

# Simple cost tracker (simulated values)

total_cost = 0.0
query_count = 0
COST_PER_QUERY = 0.0002   

# 1. get_flight_info()

def get_flight_info(flight_number: str) -> dict:
    """
    Look up a flight entry in the dataset.
    Returns dict if found, otherwise returns None.
    """
    return flight_database.get(flight_number)


# 2. info_agent_request()

def info_agent_request(flight_number: str) -> str:
    """
    Fetch raw flight info and return it as a strict JSON string.
    No additional formatting or text is added.
    """
    data = get_flight_info(flight_number)

    if data is None:
        return json.dumps({"error": "Flight not found"})

    return json.dumps(data)


# 3. qa_agent_respond()

def qa_agent_respond(user_query: str) -> str:
    """
    Parse a natural-language question for a flight number,
    retrieve related data, and format the answer as JSON.
    """

    global total_cost, query_count
    query_count += 1
    total_cost += COST_PER_QUERY
    match = re.search(r"([A-Z]{1,3}\d+)", user_query)
    if not match:
        return json.dumps({"answer": "No valid flight number detected in the query."})

    flight_number = match.group(1).upper()

    raw_json = info_agent_request(flight_number)
    info = json.loads(raw_json)

    if "error" in info:
        return json.dumps({"answer": f"No records found for flight {flight_number}."})
    final_text = (
        f"Flight {info['flight_number']} departs at {info['departure_time']} "
        f"to {info['destination']}. Current status: {info['status']}."
    )

    return json.dumps({"answer": final_text})


# Test Cases (match the evaluation requirements)

if __name__ == "__main__":

    print("\nTEST: get_flight_info('AI4669')")
    print(get_flight_info("AI4669"))

    print("\nTEST: info_agent_request('AI4669')")
    print(info_agent_request("AI4669"))

    print("\nTEST: qa_agent_respond('When does Flight IX8917 depart?')")
    print(qa_agent_respond("When does Flight IX8917 depart?"))

    print("\nTEST: qa_agent_respond('What is the status of Flight AI999?')")
    print(qa_agent_respond("What is the status of Flight AI999?"))

  # Final cost summary

    avg_cost = total_cost / query_count if query_count > 0 else 0.0
    print("\n------------------------------------")
    print(f"Total Cost available: ${total_cost:.4f}")
    print(f"Average Cost per Query: ${avg_cost:.4f}")
    print("------------------------------------")
