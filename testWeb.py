import webbrowser

url = "https://www.yorkshirefabric.com/search?page={}&q=tartan"
start, end = 3, 15
for i in range(start, end + 1):
    webbrowser.open(url.format(i))
