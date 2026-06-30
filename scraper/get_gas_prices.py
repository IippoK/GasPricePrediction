import requests
import gzip
import json
from io import BytesIO
from datetime import datetime

# Download gzipped JSON-file
url = "https://www.fuel-prices.eu/fuel_prices_2.json.gz"
response = requests.get(url)

# Extract gzip and read JSON as a string, convert to object
with gzip.GzipFile(fileobj=BytesIO(response.content)) as f:
    json_str = f.read().decode("utf-8")         # read and decode the gzip
    data = json.loads(json_str)                 # convert JSON string into Python object

# Filter for Finland
gas_prices = [entry for entry in data["data"] if entry.get("cc") == "FI"]

# Save as JSON
output = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "data": gas_prices
}

with open("gas_price_data.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
  
