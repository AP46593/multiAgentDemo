import requests, json, urllib3, ssl
urllib3.disable_warnings()              # silence the InsecureRequestWarning

url = "https://driver-vehicle-licensing.api.gov.uk/vehicle-enquiry/v1/vehicles"
payload = {"registrationNumber": "SP05WFM"}
headers = {"x-api-key": "yiLDIMmei57Cu6pAMTkrg2ftd3zPP3zQ5GV7cHbc", "Content-Type": "application/json"}

print("verify = False …")
try:
    r = requests.post(url, json=payload, headers=headers, timeout=10, verify=False)
    print("Status:", r.status_code)
    print("Body  :", r.text[:120], "…")
except Exception as e:
    print("ERROR :", e)
