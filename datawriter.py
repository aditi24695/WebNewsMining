import urllib.request as urllib2
import csv
import sqlite3 as sql
from bs4 import BeautifulSoup as bs
import feedparser

link_scrap = []
keywords = []
scrapper_link = []
description = ''
dictionary_vocab = {}
dictionary_df = {}
N = 0

class Scrapper():
# Go to the topic page of the timesofIndia and filter out all the topics and their respective RSS links
    def scrap_rss_page(self):
        page = 'https://timesofindia.indiatimes.com/rss.cms'
        page_content = urllib2.urlopen(page)
        soup = bs(page_content, "html.parser")
        div_content = soup.find_all("div", id='main-copy')
        count = 0
        for tag in div_content:
            tdtags = tag.find_all("td")
            for tag in tdtags:
                if tag.select('.rssurl'):
                    for link in tag.select('.rssurl'):
                        scrapper_link.append(link['href'])
                if tag.text:
                    if tag.text != '' and tag.text != 'More':
                        keywords.append(tag.text)
        list_key = ['Business','Sports', 'Health', 'Science', 'Education', 'World', 'Tech','India', 'Environment']
        for i in range(len(keywords)):
            if(keywords[i] in list_key):
                self.parse_rss(scrapper_link[i], keywords[i])

# parse rss_link of each page and save the required xml tags in the list news details
    def parse_rss(self,link, tag):
        d = feedparser.parse(link)
        for post in d['entries']:
            link = (post.link)
            print(link)
            description = self.beautiful_soup(link)
            print(description)
            if not description == 'abcd':
                with open('newdata.csv', 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t')
                    writer.writerow([description, tag])

    def beautiful_soup(self, each_link):
        content = 'abcd'
        try:
            page = each_link
            content=""
            page_content = urllib2.urlopen(page)
            soup = bs(page_content, "html.parser")
            content = ""
            div_content = soup.find_all('div', {'class': "article_content clearfix"})
            for wrapper in div_content:
                content = wrapper.text
            return content
        except:
            print('not good')

    def main(self):
        self.scrap_rss_page()


if __name__ == '__main__':
    webscrapper = Scrapper()
    webscrapper.main()