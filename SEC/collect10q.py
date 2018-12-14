import json
import feedparser
import re
import time
import urllib.request
import os


delay = 1

prefix = "./data/"

# with open('manifest.json') as f:
#     data = json.load(f)

with open('constituents_json.json') as f:
    indata = json.load(f)

for d in indata:
    ticker = d["Symbol"]
    # if ticker in data:
    #     print("Skipping: " + ticker)
    #     continue;
    # data[ticker] = dict()
    feed = feedparser.parse("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK="+ticker+"&type=10-Q%25&dateb=&owner=exclude&start=0&count=40&output=atom")
    if not os.path.exists(prefix+ticker):
        os.makedirs(prefix+ticker)
    for entry in feed.entries:
        link = entry.link
        summary = entry.summary
        updated = entry.updated
        link = link.replace("-index.htm",".txt")
        print("Downloading: " + link)
        filename = link.split("/")[-1]
        path = prefix+ticker+"/"+filename
        urllib.request.urlretrieve(link,path)
        # data[ticker][updated]=path
        time.sleep(delay)
    # with open('manifest.json', 'w') as outfile:
    #     json.dump(data, outfile)




