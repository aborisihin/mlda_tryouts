import random
import requests
from bs4 import BeautifulSoup


class YMSRandomUseragentMiddleware(object):
    # generates a random user-agent string from top-list
    # inspiration:
    # https://github.com/alecxe/scrapy-fake-useragent

    def __init__(self, crawler):
        super(YMSRandomUseragentMiddleware, self).__init__()

        self.user_agents = []
        self.user_agents_source = crawler.settings.get('USER_AGENT_SOURCE', None)
        if self.user_agents_source:
            req_ua = requests.get(self.user_agents_source)
            ua_parser = BeautifulSoup(req_ua.text, 'lxml')
            for ua in ua_parser.find_all('td', attrs={'class': 'useragent'}):
                self.user_agents.append(ua.find('a').text)
        else:
            self.user_agents.append(crawler.settings.get('USER_AGENT_DEFAULT', ''))

    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler)

    def process_request(self, request, spider):
        current_user_agent = random.choice(self.user_agents)
        spider.logger.info('Spider uses user-agent: {}'.format(current_user_agent))
        request.headers.setdefault('User-Agent', current_user_agent)

    def process_response(self, request, response, spider):
        return response

    def process_exception(self, request, exception, spider):
        pass
