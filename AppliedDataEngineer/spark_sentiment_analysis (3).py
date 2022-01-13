from textblob import TextBlob
import matplotlib.pyplot as plt
from pyspark import SparkConf, SparkContext
import re
import string
import translate


##OTHER FUNCTIONS/CLASSES

def resolve_emoticon(line):
   emoticon = {
    	':-)' : 'smile',
      ':('  : 'sad',
    	':))' : 'very happy',
    	':)'  : 'happy',
    	':((' : 'very sad',
    	':-P' : 'tongue',
    	':-o' : 'gasp',
    	'>:-)':'angry'
   }   
   for key in emoticon:
      line = line.replace(key, emoticon[key])
   return line

def abb_bm(line):
   abbreviation_bm = {
         'sy': 'saya',
         'sk': 'suka',
         'byk': 'banyak',
         'sgt' : 'sangat',
         'mcm' : 'macam',
         'bodo':'bodoh'
   }  
   abbrev = ' '.join (abbreviation_bm.get(word, word) for word in line.split())  
   return (resolve_emoticon(abbrev)) 

  
def abb_en(line):
   abbreviation_en = {
    'u': 'you',
    'thr': 'there',
    'asap': 'as soon as possible',
    'lv' : 'love',    
    'c' : 'see'
   } 
   abbrev = ' '.join (abbreviation_en.get(word, word) for word in line.split())
   return (resolve_emoticon(abbrev))  

def make_plot(pos,neg,neu):
  
   #This function plots the counts of positive and negative words     

   Polarity = [1,2,3]
   LABELS = ["Positive", "Negative", "Neutral"]
   Count_polarity = [int(pos), int(neg), int(neu)]

   plt.xlabel('Polarity')
   plt.ylabel('Count')
   plt.title('Sentiment Analysis - Lexical Based')

   plt.grid(True)

   plt.bar(Polarity, Count_polarity, align='center')
   plt.xticks(Polarity, LABELS)
   plt.show()
   # return (pos,neg)

def remove_features(data_str):
   
   num_re = re.compile('(\d+)')
   url_re= re.compile(r'https?://(\S+)')   
   mention_re= re.compile(r'(@|#)(\w+)')
   RT_re= re.compile(r'rt(\s+)')
   data_str = num_re.sub(' ', data_str)
   data_str= RT_re.sub(' ', data_str) # remove RT
   data_str= url_re.sub(' ', data_str) # remove hyperlinks
   data_str= mention_re.sub(' ', data_str) # remove @mentions and hash
   data_str= num_re.sub(' ', data_str) # remove numerical digit   
   # Continue to CODE IT YOURSELF

   return data_str


def main(sc,filename):
   
   mydata = sc.textFile(filename).map(lambda line: line.lower())
   mydata1 = mydata.map(lambda line:remove_features(line))
   mydata_bm = mydata1.filter(lambda line:TextBlob(line).detect_language() == 'ms').map(lambda line:abb_bm(line)).map(lambda line:line.translate('en'))
   mydata_en = mydata1.filter(lambda line:TextBlob(line).detect_language() == 'en').map(lambda line:abb_en(line))
   mydata_all = mydata_bm.union(mydata_en).map(lambda line:resolve_emoticon(line))

   sentiment_analysis = mydata_all.map(lambda line:TextBlob(line).sentiment.polarity)
   
   pos = sentiment_analysis.filter(lambda line:line > 0).count()
   neg = sentiment_analysis.filter(lambda line:line < 0).count()
   neu = sentiment_analysis.filter(lambda line:line == 0).count()
   
   print(pos, neg, neu)
   make_plot(int(pos),int(neg), int(neu)) #the cast is just to ensure the value is in integer data type
   


if __name__ == "__main__":

   # Configure your Spark environment
   conf = SparkConf().setMaster("local[*]").setAppName("My Spark Application")
   sc = SparkContext(conf=conf)
   # CODE IT YOURSELF
  
   filename = "simple_sentences.txt"
  
   main(sc, filename)

   sc.stop()
