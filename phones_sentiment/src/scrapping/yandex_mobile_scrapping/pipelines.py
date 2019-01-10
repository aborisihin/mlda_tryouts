from scrapy.exceptions import DropItem


class YMRValidatePipeline(object):
    def process_item(self, item, spider):
        if item['text'] and item['rank']:
            return item
        else:
            raise DropItem("Missing review text or rank in {}".format(item))


class YMRDuplicatesPipeline(object):
    def __init__(self):
        self.ids_seen = set()

    def process_item(self, item, spider):
        if item['id_review'] in self.ids_seen:
            raise DropItem("Duplicate item found: {}".format(item))
        else:
            self.ids_seen.add(item['id_review'])
            return item
