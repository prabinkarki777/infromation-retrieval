import scrapy
import json
from ..items import NewsItem


class NewsapiSpider(scrapy.Spider):
    name = "nytimesapi"
    allowed_domains = ["api.nytimes.com"]
    custom_settings = {
        # 12 seconds between calls to avoid hitting the per minute rate limit
        'DOWNLOAD_DELAY': 14,
        'FEEDS': {
            'ny-times-news-data.csv': {
                'format': 'csv',
                'encoding': 'utf-8',
                'fields': ['category', 'description'],
                'overwrite': True,
            }
        }
    }
    api_key = "Uw5GOLcO2Kz2FvILlZ1pd2MPs4MhV6eO"
    categories = {
        "economy": ["Financial", "Small Business", "Sunday Business", "The Business of Green", "Your Money", "DealBook", "Retail"],
        "entertainment": [
            "Movies",
            "Television",
            "Theater"
        ],
        "politics": ["politics"]
    }

    news_desk_filters = {
        category: ",".join(f'"{desk}"' for desk in desks)
        for category, desks in categories.items()
    }

    base_url = (
        "https://api.nytimes.com/svc/search/v2/articlesearch.json?"
        "api-key={api_key}&fq=news_desk:({news_desk_filter})"
        "&page={page}"
    )

    def __init__(self, *args, **kwargs):
        super(NewsapiSpider, self).__init__(*args, **kwargs)
        self.max_results_per_category = 200
        self.result_count = {category: 0 for category in self.categories}
        self.page_count = {category: 0 for category in self.categories}

    def start_requests(self):
        for category in self.categories.keys():
            url = self.base_url.format(
                api_key=self.api_key,
                news_desk_filter=self.news_desk_filters[category],
                page=0
            )
            self.logger.info(
                f"Starting requests for category '{category}' with URL: {url}")
            yield scrapy.Request(url, self.parse, meta={'category': category, 'page': 0})

    def parse(self, response):
        category = response.meta['category']
        page = response.meta['page']

        if self.result_count[category] >= self.max_results_per_category:
            return

        data = json.loads(response.text)
        docs = data.get('response', {}).get('docs', [])

        if response.status != 200:
            self.logger.error(
                f"Failed to fetch data for page {page} and category '{category}'. Status code: {response.status}"
            )
            return

        if not docs:
            self.logger.info(
                f"No more data for category '{category}' on page {page}. Ending crawl for this category.")
            return

        for article in docs:
            if self.result_count[category] >= self.max_results_per_category:
                return

            description = article.get('lead_paragraph')
            if description:
                item = NewsItem()
                item['category'] = category
                item['description'] = description
                yield item
                self.result_count[category] += 1

        # Continue to the next page if there are more results
        if self.result_count[category] < self.max_results_per_category:
            next_page = page + 1
            next_url = self.base_url.format(
                api_key=self.api_key,
                news_desk_filter=self.news_desk_filters[category],
                page=next_page
            )
            self.logger.info(
                f"Fetching next page {next_page} for category '{category}' with URL: {next_url}")
            yield scrapy.Request(next_url, self.parse, meta={'category': category, 'page': next_page})
