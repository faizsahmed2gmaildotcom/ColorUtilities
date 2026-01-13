import requests
import re
from html import unescape

url = "https://www.colorhexa.com/color-names"
response = requests.get(url)
print(f"{url} responded with code {response.status_code}")
result = re.findall('''background-color:#(.*?);"><a.*?>(.*?)</a>''', str(response.content), re.DOTALL)


def hexStrToRGB(hex_str: str) -> list[int]:
    return [int(hex_str[i: i + 2], 16) for i in range(0, 6, 2)]


for c in result:
    print(f"\"{unescape(c[1].encode('utf-8').decode('unicode_escape').encode('latin1').decode('utf-8').replace(' ', '_').lower())}\" = {hexStrToRGB(c[0])}")
