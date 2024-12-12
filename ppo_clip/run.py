from fastapi import FastAPI, Request
import uvicorn
import json
from pprint import pprint

# Create FastAPI app
app = FastAPI()

@app.post("/receive")
async def receive_data(request: Request):
# Extract and pretty print request data
    data = await request.json()
    
    # Parse the nested gameState JSON string into an object
    if 'gameState' in data:
        data['gameState'] = json.loads(data['gameState'])
    
    print("\n=== Received data ===")
    pprint(data, width=100, sort_dicts=False)
    print("===================\n")
    return {"status": "Data received", "received_data": data}

@app.get("/")
async def root():
    return {"message": "Server is running! Send POST requests to /receive."}

if __name__ == "__main__":
    # Run the app
    uvicorn.run(app, host="0.0.0.0", port=8000)
