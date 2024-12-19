import WikipediaWrapper as wiki_scrapper
import pandas as pd

df = pd.read_csv("data/raw/birds_datalist_de.csv")
print(df.head())

for index, row in df.iterrows():
    print(f'Versuche Daten zu {row["Vogelname"]} zu finden')

    scrapper = wiki_scrapper.WikipediaPageSearcher()
    article_url, thumbnail_url, page = scrapper.SearchForPage(row["Art"], row["Vogelname"])

    if not article_url:
        print(f'Zu {row["Vogelname"]} gibt es keine Informationen, skippen')
        continue

    print(article_url)

    description = scrapper.GetPageContent(page)
    print(description)

    f = open("data/raw/BirdFiles/" + row["Vogelname"] + "V1.txt", "w")
    f.write(description)
    f.close()