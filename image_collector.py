import requests
import time
from datetime import datetime

# URL of the image to download
image_url = 'https://biz.parks.wa.gov/webcams/CapedNorthHead1.jpg'

# Interval in seconds
interval = 60  # Change this to your desired interval

while True:
    # Get the current timestamp
    timestamp = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
    # Download the image
    response = requests.get(image_url)
    
    if response.status_code == 200:
        # Save the image with a timestamp
        with open(f'image_{timestamp}.jpg', 'wb') as f:
            f.write(response.content)
        print(f'Downloaded image at {timestamp}')
    else:
        print(f'Failed to retrieve image. Status code: {response.status_code}')
    
    # Wait for the specified interval before downloading again
    time.sleep(interval)