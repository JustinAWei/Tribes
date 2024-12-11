from fastapi import FastAPI, Request
import uvicorn

# Create FastAPI app
app = FastAPI()

@app.post("/receive")
async def receive_data(request: Request):
    # Extract and print request data
    data = await request.json()
    print("Received data:", data)
    return {"status": "Data received", "received_data": data}

@app.get("/")
async def root():
    return {"message": "Server is running! Send POST requests to /receive."}

if __name__ == "__main__":
    # Run the app
    uvicorn.run(app, host="0.0.0.0", port=8000)
