from textblob import TextBlob
from pyspark import SparkConf, SparkContext
import re



def abb_en(abbrev):
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
    
    if x > 0 :
        print ('positive')
    
    if x < 0 :
        print ('negative')
    
    else :
        print ('neutral')
    
    return polar
  
   
#Write your main function here
def main(sc,filename):
    myData = sc.textFile(filename).map(lambda x:x.split(",")).filter(lambda x:len(x)==8).filter(lambda x: len(x[1])>0)
    myData2 = myData.map(lambda x: x.lower()).map(lambda x: remove_features(x)).map(lambda x:abb_en(x)).map(lambda x:TextBlob(x).sentiment.polarity).map (lambda x:polarity(x))
    
    myZip = myData.zip(myData2)
    
    myClean = myZip.map(lambda x: x.remove("'")).map(lambda x:x.remove ("""))
    
    myClean.saveAsTextFile("MyFinishBitcoin")
    
   
   

  
   

if __name__ == "__main__":

 
   conf = SparkConf().setMaster("local(3)").setAppName("My bitcoin")
   sc = SparkContext(conf=conf)
   filename = "bitcoin.csv"   
   
   main (sc,filename)

   sc.stop()

