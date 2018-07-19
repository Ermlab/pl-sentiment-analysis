""" Opineo.pl company reviews scraper. """

import sqlite3
import time
from sys import argv

import requests
from lxml import html


def open_website(url):
    """ Open website and return a class ready to work on """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36'
                      ' (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'
    }
    page = requests.get(url, headers=headers)
    source = html.fromstring(page.content)
    return source


def is_verified(source):
    """ Check if review is verified or not. """
    if source.xpath(".//span[@class='revz_wo_badge w_popup_wo']"):
        return "Yes"
    else:
        return "No"


def next_page(source):
    """ Check if there is a next page with reviews. """
    try:
        next_page_url = source.xpath("//div[@class='pagi_sec_right']/a/@href")[0]
        return "http://opineo.pl{}".format(next_page_url)
    except IndexError:
        return False


def get_reviews(source):
    """ Find and return all review's data """
    reviews = source.xpath("//div[@class='revz_container']")

    for review in reviews:
        try:
            data = {
            'description': review.xpath(".//span[@itemprop='description']//text()"
                                        )[0],
            'rate': review.xpath(".//span[@itemprop='reviewRating']/strong/text()"
                                    )[0],
            'date': review.xpath(".//span[@class='revz_date']//text()"
                                    )[0].replace(" | ", ", "),
            'url': "http://opineo.pl" +
                    review.xpath(".//div[@class='revz_head']/div/a/@href")[0],
            'author': review.xpath(".//span[@class='revz_nick']//text()")[0],
            # 'author_reviews': review.xpath(".//span[@class='revz_revcount']//text()"
            #                                 )[0].split(" ")[0].replace("(", ""),
            'author_url': "http://opineo.pl" +
                            review.xpath(".//div[@class='revz_head']/div/a/@href")[0],
            'verified': is_verified(review)
            }
            yield data
        except IndexError:
            pass


def main(company_url, interval):
    """ Create Sqlite3 database and insert all found reviews. """
    if "http://" not in company_url:  # Avoid requests.exceptions.MissingSchema error.
        print("Company url must start with http:// !")
    else:
        connection = sqlite3.connect('reviews.db')
        reviews_db = connection.cursor()
        try:
            reviews_db.execute("CREATE TABLE reviews (description, rate, date, url, "
                               "author, author_url, verified)")
        except sqlite3.OperationalError:  # Table already exists.
            pass
        while True:
            print("Scraping reviews from {}".format(company_url))
            source = open_website(company_url)
            print(source)
            for data in get_reviews(source):
                #print(data)
                reviews_db.execute("INSERT INTO reviews VALUES (:description, :rate,"
                                   " :date, :url, :author, :author_url,"
                                   " :verified)", data)
                connection.commit()
            company_url = next_page(source)
            if not company_url:
                break
            time.sleep(int(3))


if __name__ == "__main__":
    urls = ["http://www.opineo.pl/opinie/aptekagemini-pl",
            "http://www.opineo.pl/opinie/apteka-melissa-pl",
            "http://www.opineo.pl/opinie/krakvet-pl",
            "http://www.opineo.pl/opinie/wapteka-pl",
            "http://www.opineo.pl/opinie/zooart-com-pl",
            "http://www.opineo.pl/opinie/empik-com",
            "http://www.opineo.pl/opinie/north-pl",
            "http://www.opineo.pl/opinie/aros-pl"
            ]
    for url in urls:
        try:
            main(url, argv[1])
        except BaseException:
            print("Error at {}, continue...".format(url))
            continue
