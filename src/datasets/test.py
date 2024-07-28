import requests

url = "https://lecture.emind.vn/api/lecture/generate"

payload = {'user_name': 'Nguyen Vu',
           'user_email': 'nguyenvuhoang9712@gmail.com'}
files = [
    ('file', ('L5_Ensembling_v2.pdf', open(
        '/Users/nguyen/Downloads/L5_Ensembling_v2.pdf', 'rb'), 'application/pdf'))
]
headers = {}

for i in range(100):
  response = requests.request(
      "POST", url, headers=headers, data=payload, files=files)

  print(response.text)
