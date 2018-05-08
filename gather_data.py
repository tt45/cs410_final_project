import quandl
import json
import datetime
import requests
import csv

def get_bitcoin_price():
    data = quandl.get("BCHARTS/BITSTAMPUSD", authtoken="u4bnPE8zeaFZxMBNne6V", start_date="2016-11-01", end_date="2018-05-01")
    #print (data[['Open', 'Close', 'Weighted Price']])
    bitcoin_price_data = {}
    date_iter=datetime.date(2018,5,1)
    for i in range(len(data['Weighted Price'])):
        str_date = str(date_iter).replace("-", "")
        bitcoin_price_data[str_date] = data['Weighted Price'][str(date_iter)]
        date_iter -= datetime.timedelta(days=1)
    return bitcoin_price_data


def get_sentiment():
    date = datetime.date(2018, 5, 1)
    sentiment_score = {}
    while (date.year!=2016 or date.month!=10 or date.day!=31):

        month_string = str(date.month)
        if len(month_string) < 2:
            month_string = "0" + month_string
        day_string = str(date.day)
        if len(day_string) < 2:
            day_string = "0" + day_string
        date_string = str(date.year) + month_string + day_string

        r1 = requests.get("http://archive.org/wayback/available?url=reddit.com/r/bitcoin&timestamp=" + date_string)

        if(r1.status_code == 200):
            data1 = json.loads(r1.text)
            archive_url = data1['archived_snapshots']['closest']['url']
        else:
            archive_url = None
            print("Error return code = "+str(r1.status_code))

        r2 = requests.get("https://api.havenondemand.com/1/api/sync/analyzesentiment/v2?apikey=07af18eb-e943-4dd0-9fdf-93b06614c921&url=" + archive_url, verify = False, timeout = 100)
        if(r2.status_code == 200):
            data2 = json.loads(r2.text)
            #sentiment_score['date'] =
            sentiment_score[date_string] = data2['sentiment_analysis'][0]['aggregate']['score']
        else:
            print("Error return code = "+str(r2.status_code))

        date -= datetime.timedelta(days=1)

    return sentiment_score


def merge_data(price_data, sentiment_data):
    date_iter = datetime.date(2018, 5, 1)
    myData = [['Date', 'Price', 'Sentiment']]
    myFile = open('merged_data.csv', 'w+')
    with myFile:
        writer = csv.writer(myFile)
        for i in range(len(price_data)):
            str_date = str(date_iter).replace("-", "")
            writer.writerow([str_date, price_data[str_date], sentiment_data[str_date]])
            date_iter -= datetime.timedelta(days=1)


def main():

    bitcoin_price={}
    with open('bitcoin_price.json') as f:
        bitcoin_price = json.load(f)
    sentiment_score={}
    with open('sentiment_score.json') as f:
        sentiment_score = json.load(f)
    merge_data(bitcoin_price,sentiment_score)

if __name__ == '__main__':
    main()
