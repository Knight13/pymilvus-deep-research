import wikipediaapi

class WikipediaFetcher:
    def __init__(self, language: str = "en", user_agent: str = "MilvusDeepResearchBot (<insert your email>)"):
        self.wiki = wikipediaapi.Wikipedia(language=language, user_agent=user_agent)

    def get_page(self, page_title: str):
        return self.wiki.page(page_title)
