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

def polarity(polar):
    
    if polar > 0:
        return ("Positive")
    elif polar < 0:
        return ("Negative")
    else:
        return ("Neutral")
   
  
   
#Write your main function here
def main(sc,filename):
   
    RDD1 = (sc.textFile(filename).map(lambda x:x.split(",")).filter(lambda x:len(x) == 8).filter(lambda x:len(x[1])>0))
    RDD2 = RDD1.map(lambda x:x[7]).map(lambda x:remove_features(x)).map(lambda x:x.lower()).map(lambda x:abb_en(x)).map(lambda x:TextBlob(x).sentiment.polarity).map(lambda x:polarity (x))
    
    RDD3 = RDD1.map(lambda x:','.join(x)).zip(RDD2).map(lambda x:x[0]+','+x[1])
    RDD4 = RDD3.map(lambda x:x.replace ("'",'').replace('"',''))
    
    final = RDD4.saveAsTextFile("bitcoin1")
   

  
   

if __name__ == "__main__":
    
    conf = SparkConf().setMaster("local[1]").setAppName("My Bitcoin Application")
    sc = SparkContext(conf = conf)
    filename = "bitcoin.csv"
    main(sc,filename)

    sc.stop()
