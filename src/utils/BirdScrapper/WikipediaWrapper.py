import requests
import json

class WikipediaPageSearcher():
    def __init__(self):
        pass

    def SearchForPage(self, searchString, searchStringAlternative):
        language_code = 'de' # Deutsch
        search_query = 'Vogel ' + searchString
        number_of_results = 1

        headers = {
        'Authorization': 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiIwZTYwODc4OGNmYWIwNjE2NzkyYTFlZjhhMjgwOTE1MSIsImp0aSI6Ijc5OGEyNzlmYzFlMmIyYTZiNjlkN2IwOTAxZGJlMjFiZmE0ZjczOWNkNjkzNTc2MmQ0MTU4ZDU3YzRkNmNmNWNiZDk2ZGRhOGI2OGRlMjE4IiwiaWF0IjoxNzM0NjM4NjY5LjU0OTkwMSwibmJmIjoxNzM0NjM4NjY5LjU0OTkwNCwiZXhwIjozMzI5MTU0NzQ2OS41NDczMzMsInN1YiI6Ijc3MjE0OTMwIiwiaXNzIjoiaHR0cHM6Ly9tZXRhLndpa2ltZWRpYS5vcmciLCJyYXRlbGltaXQiOnsicmVxdWVzdHNfcGVyX3VuaXQiOjUwMDAsInVuaXQiOiJIT1VSIn0sInNjb3BlcyI6WyJiYXNpYyJdfQ.RotolPlXwRdRJr5T_X6iOwmcNuLayBoduj9blFovkiH2zRTurfTrNM_6WBpMHuuZ5y3qQwCntsttDP8vSL_-9FzMpE1xMjfk3nCu6-MK-vjoiAaC8CkXyR-daKbMjk4CdnEKwYztJaV_g1it4gvTo_voGgPhdMAJXfVLdzjhQEW9Dq5MSQfMIfiue08M7iXzBHMpVx82clq4aIkDDmdWfx1Y8_mYX3R1iCMx4K2Tqbbmcd9MfkHXCWH51mZxAfYa1m9qEG1hFng7k7GGvjyfyEQ7Ru6hj3tYxq-jKvI_voWOqKpcCEpVTF0Xdd5VvURfGgZPG_0KCX1TKRwVQh89TFVIMCUM99CD_d9ntfR6KHCJIXHKelOAz28H9lWlyZ_6dbehLSwNvpfRIKH0XzISxiPWnLMNwxzpZ952sedY842nRBEmw8cAwRy-lMTP1LZ2YqgKGTr7t8PDin-2AVhPs9Wfk8vhjlavvNPudBST-GUwTrV7QiEYwc745BnFUw6F0-N9mQV96Ob0ZqKblZ7bx1vDpEps2u8esQGW21peekLXaTGpZviVK5GC2eIMecUyFdt22-AuXneZuz7GLjIM1DR_Gzo3Q27ATlJ8xB2f7-kvImep3UGHfBPXjrBAokFt5o-4uLzMQ4mu4kmxzrHKGSPpv1MkOFiarMJUA418WbU',
        'User-Agent': 'BFH Projekt'
        }

        base_url = 'https://api.wikimedia.org/core/v1/wikipedia/'
        endpoint = '/search/page'
        url = base_url + language_code + endpoint
        parameters = {'q': search_query, 'limit': number_of_results}
        response = requests.get(url, headers=headers, params=parameters)

        # Get article title, description, and URL from the search results

        response = json.loads(response.text)

        if not response['pages']:
            parameters = {'q': searchStringAlternative, 'limit': number_of_results}

            response = requests.get(url, headers=headers, params=parameters)
            response = json.loads(response.text)

            if not response['pages']:
                return None, None, None

        for page in response['pages']:
            display_title = page['title']
            article_url = 'https://' + language_code + '.wikipedia.org/wiki/' + page['key']
            try:
                article_description = page['description']
                print(article_description)
            except:
                article_description = 'a Wikipedia article'
            try:
                thumbnail_url = 'https:' + page['thumbnail']['url']
            except:
                thumbnail_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/80/Wikipedia-logo-v2.svg/200px-Wikipedia-logo-v2.svg.png'

        return article_url, thumbnail_url, page['key']
    
    def GetPageContent(self, page):
        
        # Deutsch
        # Define the URL for the Wikipedia API
        url = "https://de.wikipedia.org/w/api.php"

            # Set up the parameters for the API request
        params = {
        "action": "query",          # Action to perform
        "format": "json",           # Format of the response
        "titles": page,             # Title of the Wikipedia page
        "prop": "extracts",         # Property to fetch (extracts of the page)
        "explaintext": True,         # Get plain text (without HTML)
        "redirects": 1              # Follow redirects
        }

        # Make the API request
        response = requests.get(url, params=params)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            page = next(iter(data['query']['pages'].values()))  # Get the first page
            if 'extract' in page:
                return page['extract']  # Return the content
            else:
                return "Page not found."
        else:
            return "Error fetching data."
        
if __name__ == "__main__":
    wiki_searcher = WikipediaPageSearcher()
    article_url, thumbnail_url, page = wiki_searcher.SearchForPage("Star")
    print(article_url)

    htmlContent = wiki_searcher.GetPageContent(page)
    print(htmlContent)