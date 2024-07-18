import scrapy
import json
from ..items import NewsItem


class NewsapiSpider(scrapy.Spider):
    name = "newsapi"
    allowed_domains = ["newsapi.org"]
    custom_settings = {
        'DOWNLOAD_DELAY': 1,
        'FEEDS': {
            'newsdata.csv': {
                'format': 'csv',
                'encoding': 'utf-8',
                'fields': ['description', 'category'],
                'overwrite': True,
            }
        }
    }
    categories = ["sports", "business", "entertainment", "health", "politics"]
    start_urls = [
        "https://newsapi.org/v2/top-headlines?country=us&apiKey=f5d17a6710fd4349a608f88417d54eaf&pageSize=100&category={category}"
    ]

    def __init__(self):
        self.category_limits = {category: 0 for category in self.categories}
        self.max_results_per_category = 50

    def start_requests(self):
        for category in self.categories:
            url = self.start_urls[0].format(category=category)
            yield scrapy.Request(url, self.parse, meta={'category': category})

    def parse(self, response):
        data = json.loads(response.text)
        category = response.meta['category']
        for article in data.get('articles', []):
            if self.category_limits[category] >= self.max_results_per_category:
                continue
            description = article.get('description')

            if description:
                item = NewsItem()
                item['description'] = description
                item['category'] = category
                self.category_limits[category] += 1

                yield item
