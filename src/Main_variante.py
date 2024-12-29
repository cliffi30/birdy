# Pipeline welche die verschiedenen Textdateien zu jedem Vogel aus dem Datenordner liest und
# daraus dann Embeddings produziert. Die Embeddings werden in der ChromaDb mit der Erweiterung von langChain
# direkt abgespeichert.
# Muss nur 1x ausgeführt werden. Danach kann mit Main_Query.py direkt die Abfrage gemacht werden.
def main():
    build_birds_embeddings(use_chrome_embeddings=True)


if __name__ == "__main__":
    main()