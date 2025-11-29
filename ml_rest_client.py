import json 
import requests

url = 'https://keva-lifeful-semichemically.ngrok-free.dev/predict'

request_data = json.dumps({'age':40, 'salary': 500000})
response = requests.post(url, request_data)
print(response.text)