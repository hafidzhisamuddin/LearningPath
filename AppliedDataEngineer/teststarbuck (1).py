from textblob import TextBlob
from pyspark import SparkConf, SparkContext
import re



def abb_en(line):
   abbreviation_en = {
    'u': 'you',
    'thr': 'there',
    'asap': 'as soon as possible',
    'lv' : 'love',    
    'c' : 'see'
   }
   
   abbrev = ' '.join (abbreviation_en.get(word, word) for word in line.split())
   return (abbrev)

def remove_features(data_str):
   
    url_re = re.compile(r'https?://(www.)?\w+\.\w+(/\w+)*/?')    
    mention_re = re.compile(r'@|#(\w+)')  
    RT_re = re.compile(r'RT(\s+)')
    num_re = re.compile(r'(\d+)')
    
    data_str = str(data_str)
    data_str = RT_re.sub(' ', data_str)  
    data_str = data_str.lower()  
    data_str = url_re.sub(' ', data_str)   
    data_str = mention_re.sub(' ', data_str)  
    data_str = num_re.sub(' ', data_str)
    return data_str

def polarity(x):
    if x >0:
        return ("+ve")
    elif x <0:
        return("-ve")
    else:
        return("neu")
   
  
   
#Write your main function here
def main():
   
    a = sc.textFile("teststarbuck.csv").map(lambda x:x.split(","))\
    .filter(lambda x:len(x[1])>0).filter(lambda x:len(x)==9)
    
    b = a.map(lambda x:x[5]).map(lambda x:remove_features(x))\
    .map(lambda x:abb_en(x)).map(lambda x:x.lower())\
    .map(lambda x:TextBlob(x).sentiment.polarity).map(lambda x:polarity(x))
    
    c = a.map(lambda x:','.join(x)).zip(b).map(lambda x:x[0]+','+x[1])\
    .map(lambda x:x.replace("'",'').replace('"',''))
    
    d = c.map(lambda x:x.split(",")).map(lambda x:(x[0],x[5],x[2],x[1],x[9],x[4],x[6],x[7],x[8]))
    
    print(d.take(5))
    #d = c.saveAsTextFile("bitcoin3")

  #export PYSPARK_DRIVER_PYTHON=/opt/anaconda3/bin/python
   

if __name__ == "__main__":
   
    conf = SparkConf().setMaster("local[1]").setAppName("My Spark Bitcoin")
    sc = SparkContext(conf=conf)
    
    filename = "teststarbuck.csv"
    
    main()
   
    sc.stop()
