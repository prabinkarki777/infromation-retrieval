import scrapy
from ..items import PublicationDetailItem, AuthorItem


class SearchEngineSpiderSpider(scrapy.Spider):
    name = "search_engine_spider"
    allowed_domains = ["pureportal.coventry.ac.uk"]
    start_urls = [
        "https://pureportal.coventry.ac.uk/en/organisations/eec-school-of-computing-mathematics-and-data-sciences-cmds/publications/"]

    def parse(self, response):
        # Publication Loop
        for result in response.css('li.list-result-item '):
            publication_link = result.css('h3.title a::attr(href)').get()
            yield response.follow(publication_link, self.parse_publication)

    def parse_publication(self, response):
        item = PublicationDetailItem()
        item['title'] = response.css(
            'div.introduction div.rendering h1 span::text').get()

        # Extract authors with links
        authors_with_link = response.css('p.relations.persons a.link.person')
        item['authors'] = self.extract_authors(authors_with_link)

        # Extract non-linked authors
        non_linked_authors = response.css('.relations.persons::text').getall()
        item['authors'].extend(
            self.extract_non_linked_authors(non_linked_authors))

        # Extract year
        item['publication_year'] = response.css(
            'tr.status span.date::text').get()

        # Extract publication_link
        item['publication_link'] = response.url

        yield item

    def extract_authors(self, authors_with_link):
        authors = []
        for author in authors_with_link:
            author_item = AuthorItem()
            author_item['name'] = author.css('span::text').get()
            author_item['profile_link'] = author.attrib['href']
            authors.append(dict(author_item))
        return authors

    def extract_non_linked_authors(self, non_linked_authors):
        authors = []
        for author_text in non_linked_authors:
            names = [name.strip(', ') for name in author_text.split(
                ',') if name.strip(', ')]
            for name in names:
                author_item = AuthorItem()
                author_item['name'] = name
                authors.append(dict(author_item))
        return authors
