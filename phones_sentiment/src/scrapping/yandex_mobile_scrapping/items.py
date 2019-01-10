import scrapy


class YandexMobileReviewItem(scrapy.Item):
    id_review = scrapy.Field(serializer=str)
    model = scrapy.Field(serializer=str)
    text = scrapy.Field(serializer=str)
    rank = scrapy.Field(serializer=int)

    def __repr__(self):
        return repr({'id': self['id_review']})
