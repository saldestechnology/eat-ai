import re

import requests

base_url = "https://www.thefork.com/restaurants/stockholm-c528294"  # from https://www.thefork.com/robots.txt
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:98.0) Gecko/20100101 Firefox/98.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0"
}

for i in range(1, 19):
    url = f"{base_url}?p={i}"
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        raise IOError(f"Got status: {resp.status_code} for {url}")
    urls = re.findall("https://www.thefork.com/restaurant/[\w-]+", resp.text)
    if len(url) == 0:
        raise ValueError(f"No restaurant urls in {url} source")
    print(urls)
