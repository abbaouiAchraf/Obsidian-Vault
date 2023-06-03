## Text Tricks
![[Pasted image 20230307231517.png]]
![[Pasted image 20230307231719.png]]
![[Pasted image 20230307232102.png]]
![[Pasted image 20230308005259.png]]
![[Pasted image 20230308212113.png]]
![[Pasted image 20230308212151.png]]
## Regular Expressions
[Regular expressions documentation in Python 3](https://docs.python.org/3/library/re.html)

## Tips and tricks of the trade for cleaning text in Python
[https://stanford.edu/~rjweiss/public_html/IRiSS2013/text2/notebooks/cleaningtext.html](https://stanford.edu/~rjweiss/public_html/IRiSS2013/text2/notebooks/cleaningtext.html)
[https://www.analyticsvidhya.com/blog/2014/11/text-data-cleaning-steps-python/](https://www.analyticsvidhya.com/blog/2014/11/text-data-cleaning-steps-python/)
[http://ieva.rocks/2016/08/07/cleaning-text-for-nlp/](http://ieva.rocks/2016/08/07/cleaning-text-for-nlp/)
[https://chrisalbon.com/python/cleaning_text.html](https://chrisalbon.com/python/cleaning_text.html)





## NLTK : Natural Language ToolKit.
### Importing :
![[Pasted image 20230319160205.png]]
### Frequency of words : 
![[Pasted image 20230319160810.png]]
### Normalization and Stemming : 
**Normalization** is finding different form of the same *word*.
**Stemming** is bring a word to its base form.
```
input = "List listed lists listing listings"
words = input.lower().split(' ')
# lower() to unify the capital and lower case words
porter = nltk.PorterStemmer()
stem_words = [poster.stem(t) for t in words]
# it ll return us [u'list',u'list',u'list',u'list',u'list'] the base form of all the words in our case its the same base form
```
### Lemmatization : 
**Lemmatization**  witches any kind of a word to its base root mode. Lemmatization is responsible for grouping different inflected forms of words into the root form, having the same meaning.(Its Stemming but resulting stems are all valid words; keeping the meaning)
```
WNlemma = nltkWordNetLemmatizer()
Lemma_words = [WNlemmalemmatize(t) for t in listWords]
```
### Tokenization : 
![[Pasted image 20230319163328.png]]
it bring the *n't* as a token which is representation or negation which is great
### Sentence Splitting :
Split text from a long text string by using nltk
```
text = "This is the first sentence. A gallon of milk in the U.S. costs $2.99. Is this the third sentences? Yes, it is!"
sentences = nltk.sent_tokenize(text)
len(sentences)
>> 4 
sentences
>>['This is the first sentence.','A gallon of milk in the U.S. costs $2.99.','Is this the third sentences?','Yes, it is!']
```




## Advanced NLTK :
### Part-of-speech Tagging :
![[Pasted image 20230326131014.png]]
```
import nltk
ntlk.help.upenn_taggset('MD')
# it ll give you information about the tag and exemples also
```
**Performing POS tagging with nltk : **
- Splitting sentence into tokens .
- Run pos-tagger .
```
ntlk.pos-tag(text_token)
>> list of tuples with (token,tag)
```
With pos-tagging we can know the relation between the words
### Parsing Sentence Structure :
![[Pasted image 20230326132802.png]]
**Ambuguity in Parsing**
![[Pasted image 20230326132955.png]]

### Spelling Correction :

This process requires dictionary of valid words, NTLK provides a solution : *words* from *nltk.corpus*
A way to me measure spelling similarity (Edit distance) between two strings.
- Jaccard index / coefficient of similarity :
![[Pasted image 20230326151123.png]]
-> what is the valid word that have the min distance or max similarity with the misspelled word and use that as alternative.
 