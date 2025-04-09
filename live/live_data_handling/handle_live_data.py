import json
import time
import pandas as pd
from threading import Thread
from queue import Queue

class HandleLiveData():
    def __init__(self):
        # Shared queue for data passing
        self.data_queue = Queue(maxsize=1000)

        # In-memory buffer for recent data
        self.recent_data = []  # Could use a pandas DataFrame or deque with max length

    def process_incoming_data(self, message):
        """Process incoming market data messages"""
        try:
            data = json.loads(message)
            
            # Extract data from both bid and ask messages
            if data["payload"]["priceType"] == "bid":
                bid_data = data["payload"]
                # Store temporarily until matching ask arrives
                # (In a real system, you'd need to handle matching more robustly)
            elif data["payload"]["priceType"] == "ask":
                ask_data = data["payload"]
                
                # Calculate midpoint values
                midpoint = {
                    "epic": bid_data["epic"],
                    "resolution": bid_data["resolution"],
                    "t": bid_data["t"],
                    "o": (bid_data["o"] + ask_data["o"]) / 2,
                    "h": (bid_data["h"] + ask_data["h"]) / 2,
                    "l": (bid_data["l"] + ask_data["l"]) / 2,
                    "c": (bid_data["c"] + ask_data["c"]) / 2,
                    "spread": ask_data["c"] - bid_data["c"]  # Optional but useful
                }
                
                # Add to queue for processing
                self.data_queue.put(midpoint)
                
        except Exception as e:
            print(f"Error processing message: {e}")

    def data_processor(self):
        """Consumer thread that processes the midpoint data"""
        while True:
            midpoint_data = self.data_queue.get()
            
            # Add to in-memory buffer
            self.recent_data.append(midpoint_data)
            
            # If buffer grows too large, trim it
            if len(self.recent_data) > 1000:  # Keep last 1000 points
                self.recent_data.pop(0)
            
            # Periodically save to disk (every N points or time interval)
            if len(self.recent_data) % 100 == 0:  # Every 100 data points
                save_to_file()
            
            # Feed to model for prediction
            prediction = model.predict(recent_data)
            
            # Act on prediction if confidence is high enough
            if prediction['confidence'] > threshold:
                execute_trade(prediction)
                
            data_queue.task_done()

    def save_to_file():
        """Save recent data to disk periodically"""
        df = pd.DataFrame(recent_data)
        df.to_csv('market_data.csv', mode='a', header=False, index=False)

    # Start processor thread
    processor_thread = Thread(target=data_processor, daemon=True)
    processor_thread.start()

    # Main thread receives data
    while True:
        message = receive_market_data()  # Your data subscription method
        process_incoming_data(message)