"""
This file defines the UMLS_API class, which provides a simple interface for interacting with the UMLS (Unified Medical Language System) database.
Acknowledgement: Codes modified from https://github.com/ruiyang-medinfo/KG-Rank
"""

import requests
from tqdm import tqdm


class UMLS_API:
    def __init__(self, apikey, version="current"):
        self.apikey = apikey
        self.version = version
        self.search_url = f"https://uts-ws.nlm.nih.gov/rest/search/{version}"
        self.content_url = f"https://uts-ws.nlm.nih.gov/rest/content/{version}"
        self.content_suffix = "/CUI/{}/{}?apiKey={}"

    def search_cui(self, query):
        """search cui of the query."""
        cui_results = []

        try:
            page, size = 1, 1
            query = {"string": query, "apiKey": self.apikey, "pageNumber": page, "pageSize": size}
            r = requests.get(self.search_url, params=query)  # HTTP request
            r.raise_for_status()
            r.encoding = 'utf-8'
            outputs = r.json()

            items = outputs["result"]["results"]
            if len(items) == 0:
                print("No results found.\n")
            for result in items:
                cui_results.append((result["ui"], result["name"]))  # return the cui and name of keywords in query.

        except Exception as except_error:
            print(except_error)

        return cui_results

    def get_concepts(self, cui):
        """get concepts of the cui."""
        print("*****UMLS - Getting Concepts*****")
        try:
            concept_suffix = "/CUI/{}?apiKey={}"
            suffix = concept_suffix.format(cui, self.apikey)
            r = requests.get(self.content_url + suffix)
            r.raise_for_status()
            r.encoding = "utf-8"
            outputs = r.json()

            return outputs["result"]
        except Exception as except_error:
            print(except_error)

    def get_semantic_types(self, cui):
        """get semantic types of the cui."""
        res = self.get_concepts(cui)
        if res is None:
            return None
        semantic_types = res["semanticTypes"]
        return semantic_types

    def get_definitions(self, cui):
        """get definitions of the cui."""
        print("*****UMLS - Getting Definitions*****")
        try:
            suffix = self.content_suffix.format(cui, "definitions", self.apikey)
            r = requests.get(self.content_url + suffix)
            r.raise_for_status()
            r.encoding = "utf-8"
            outputs = r.json()

            return outputs["result"]
        except Exception as except_error:
            print(except_error)

    def get_relations(self, cui, pages=20):
        """get relations of the cui."""
        print("*****UMLS - Getting Relations*****")
        all_relations = []

        try:
            for page in tqdm(range(1, pages + 1), desc="Getting relations in pages..."):
                suffix = self.content_suffix.format(cui, "relations", self.apikey) + f"&pageNumber={page}"
                r = requests.get(self.content_url + suffix)
                r.raise_for_status()
                r.encoding = "utf-8"
                outputs = r.json()

                page_relations = outputs.get("result", [])
                all_relations.extend(page_relations)

        except Exception as except_error:
            print(except_error)

        return all_relations
