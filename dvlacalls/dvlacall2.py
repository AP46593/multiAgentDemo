import requests, json

url = "https://driver-vehicle-licensing.api.gov.uk/vehicle-enquiry/v1/vehicles"
headers = {
    "x-api-key": "yiLDIMmei57Cu6pAMTkrg2ftd3zPP3zQ5GV7cHbc",
    "Content-Type": "application/json",
    "Accept": "application/json",
}
payload = json.dumps({"registrationNumber": "SP05WFM"})

#r = requests.post(url, headers=headers, data=payload, timeout=10)
r = requests.post(url, headers=headers, data=payload, verify=False)
print(r.status_code, r.text)