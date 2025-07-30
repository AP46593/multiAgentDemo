import requests
 
# Replace with your actual DVLA API key
API_KEY = "yiLDIMmei57Cu6pAMTkrg2ftd3zPP3zQ5GV7cHbc"
 
# Replace with the vehicle registration number you want to look up
registration_number = "SP05WFM"
 
# DVLA Vehicle Enquiry API endpoint
url = "https://driver-vehicle-licensing.api.gov.uk/vehicle-enquiry/v1/vehicles"
 
# Headers
headers = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}
 
# Payload
payload = {
    "registrationNumber": registration_number
}
 
# Make POST request
response = requests.post(url, json=payload, headers=headers)
 
# Handle response
if response.status_code == 200:
    data = response.json()
    print("Vehicle Information:")
    for key, value in data.items():
        print(f"{key}: {value}")
else:
    print(f"Error: {response.status_code} - {response.text}")