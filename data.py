

path = 'G://Il mio Drive/Datasets/artpedia/artpedia.json'
# Avoids 403 Unauthorized from website
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'}

with open(pl.Path(path),'r') as f:
    data = json.load(f)


# prepare image + question
url = v['img_url']
response = requests.get(url,headers = headers)
with Image.open(io.BytesIO(response.content)) as im:
    imgplot = plt.imshow(im)
    plt.show()
text = "How many cats are there?"
