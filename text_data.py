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
    # Creates Database Connection
    def create_connection(self):
        database = 'C:\\Users\\aditi\\OOAD\\db.sqlite3'
        return sql.connect(database)


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
        news_details = []
        for post in d['entries']:
            link = (post.link)
            description = self.beautiful_soup(link)
            with open("data.csv", "w+",encoding = 'utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(description)
            #news_details.append([post.title, descrip1tion, post.link, post.published, tag, 0])
            global N
            N = N+1
        #self.store(news_details);


# from the news_details list store the required topics from xml in the db
    def store(self,news_details):
        con = self.create_connection();
        c = con.cursor()
        c.executemany('INSERT INTO newsapp_topstories (title,desc,link,published,tag,status) VALUES (?,?,?,?,?,?);',
                      news_details)
        con.commit()
        con.close()

    def beautiful_soup(self, each_link):
        page = each_link
        content=""
        try:
            page_content = urllib2.urlopen(page)
        except Exception as e:
            print(page, "Something Bad Happened !")
            return ""
        else:
            soup = bs(page_content, "html.parser")
            content = ""
            div_content = soup.find_all('div', {'class': "article_content clearfix"})
            for wrapper in div_content:
                content = wrapper.text
        return content


    def main(self):
        self.scrap_rss_page()


if __name__ == '__main__':
    webscrapper = Scrapper();
    webscrapper.main()