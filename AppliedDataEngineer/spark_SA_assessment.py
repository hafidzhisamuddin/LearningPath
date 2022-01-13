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



def polarity (polar):
    
    if polar > 0:
        return ("+ve")
    elif polar < 0:
        return ("-ve")
    else:
        return ("neu")
  
   
#Write your main function here
def main(sc,filename):
    
    myData = (sc.textFile(filename).map(lambda x:x.split(",")).filter(lambda x:len(x)==8).filter(lambda x:len(x[1])>0))
    myData2 = myData.map(lambda x:x[4]).map(lambda x:x.lower()).map(lambda x:remove_features(x)).map(lambda x:abb_en(x)).map(lambda x:TextBlob(x).sentiment.polarity).map(lambda x:polarity(x))
    myData3 = myData.map(lambda x:','.join(x)).zip(myData2).map(lambda x:x[0]+','+x[1]).map(lambda x:x.replace("'",'').replace('"',''))
    myData4 = myData3.map(lambda x:x.split(",")).map(lambda x:(x[8],x[0],x[4],x[2],x[1],x[3],x[5],x[6],x[7]))
                                                                      
    #print (myData4.take(5))
    
    myStarbuck = myData4.saveAsTextFile("Starbuck_Saya")
   
   

  
   

if __name__ == "__main__":

   conf = SparkConf().setMaster("local[1]").setAppName("My Starbuck")
   sc = SparkContext (conf=conf)
   filename="starbucks_v1.csv"
   main(sc,filename)

   sc.stop()
