# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class Street(scrapy.Item):
    uf = scrapy.Field()
    city = scrapy.Field()
    name = scrapy.Field()
    latitude = scrapy.Field()
    longitude = scrapy.Field()
    filename = scrapy.Field()
    pano_id = scrapy.Field()
    direction = scrapy.Field()