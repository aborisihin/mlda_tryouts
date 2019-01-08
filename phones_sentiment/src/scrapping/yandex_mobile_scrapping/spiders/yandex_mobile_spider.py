import scrapy

from yandex_mobile_scrapping.items import YandexMobileReviewItem


class YandexMobileSpider(scrapy.Spider):
    name = 'yandex_mobile'
    allowed_domains = ['market.yandex.ru']

    def __init__(self, start_page=None, *args, **kwargs):
        self.start_urls = [start_page]

    def parse(self, response):
        if response.status not in [200]:
            raise scrapy.exceptions.CloseSpider('response error')
            return

        # get links to phone model reviews pages
        reviews_pages = response.xpath('//a[contains(@class, "{}")]/@href'.format('n-snippet-card2__rating')).extract()
        for r_page in reviews_pages:
            yield response.follow(r_page, callback=self.parse_reviews_page)

        # get link to the next page with phone models
        next_page = response.xpath('//a[contains(@class, "n-pager__button-next")]/@href').extract_first()
        if next_page is not None:
            yield response.follow(next_page, callback=self.parse)

    def parse_reviews_page(self, response):
        if response.status not in [200]:
            raise scrapy.exceptions.CloseSpider('response error')
            return

        # scrap reviews
        model = response.xpath('//meta[@itemprop="name"]/@content').extract_first()
        review_blocks = response.xpath('//div[contains(@class, "n-product-review-item i-bem")]')
        for review in review_blocks:
            id_review = review.xpath('@data-review-id').extract_first()
            text_blocks = review.xpath('*/dd[contains(@class, "n-product-review-item__text")]/text()')
            text = ' '.join([tb.extract() for tb in text_blocks])
            rank = review.xpath('*/*/div[contains(@class, "rating__value")]/text()').extract_first()
            yield YandexMobileReviewItem(id_review=id_review, model=model, text=text, rank=rank)

        # get link to the next page with reviews
        next_page = response.xpath('//a[contains(@class, "n-pager__button-next")]/@href').extract_first()
        if next_page is not None:
            yield response.follow(next_page, callback=self.parse_reviews_page)
