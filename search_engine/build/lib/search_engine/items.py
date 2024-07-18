# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class PublicationDetailItem(scrapy.Item):
    title = scrapy.Field()
    authors = scrapy.Field()
    publication_year = scrapy.Field()
    publication_link = scrapy.Field()


class AuthorItem(scrapy.Item):
    name = scrapy.Field()
    profile_link = scrapy.Field()


class NewsItem(scrapy.Item):
    description = scrapy.Field()
    category = scrapy.Field()
