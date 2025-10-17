# /// script
# dependencies = [
#   "requests",
#   "beautifulsoup4",
# ]
# ///
#
# Extract N320 Reduced Gaussian Grid from ECMWF documentation
# Source: https://www.ecmwf.int/en/forecasts/documentation-and-support/gaussian_n320

import csv

import requests
from bs4 import BeautifulSoup  # type: ignore[import-untyped]

# Download and parse the table from ECMWF
print("Downloading N320 grid table from ECMWF...")
url = "https://www.ecmwf.int/en/forecasts/documentation-and-support/gaussian_n320"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

# Find and parse the table
table = soup.find("table")
rows = table.find_all("tr")

data = []
for row in rows[1:]:  # Skip header row
    cols = row.find_all("td")
    if len(cols) >= 5:
        nlon = int(cols[1].text.strip())  # Reduced Points (Standard)
        lat = float(cols[4].text.strip())  # Latitude
        data.append((lat, nlon))

print(f"Extracted {len(data)} rows from table")
print(f"Total points: {sum(nlon for _, nlon in data)}")

# Write to CSV file
csv_path = "src/xpublish_tiles/testing/grids/n320_grid.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["latitude", "num_points"])
    for lat, nlon in data:
        writer.writerow([lat, nlon])

print(f"Wrote {len(data)} rows to {csv_path}")
