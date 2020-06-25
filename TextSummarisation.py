import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.porter import *
import tensorflow_hub as hub
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import re
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import heapq

#nltk.download('punkt')
#nltk.download('stopwords')
contractions = { 
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}

passage = """
Game of Thrones is an American fantasy drama television series created by David Benioff and D. B. Weiss for HBO. It is an adaptation of A Song of Ice and Fire, George R. R. Martin's series of fantasy novels, the first of which is A Game of Thrones. The show was both produced and filmed in Belfast and elsewhere in the United Kingdom. Filming locations also included Canada, Croatia, Iceland, Malta, Morocco, and Spain. The series premiered on HBO in the United States on April 17, 2011, and concluded on May 19, 2019, with 73 episodes broadcast over eight seasons.

Set on the fictional continents of Westeros and Essos, Game of Thrones has a large ensemble cast and follows several story arcs. One arc is about the Iron Throne of the Seven Kingdoms of Westeros and follows a web of alliances and conflicts among the noble dynasties either vying to claim the throne or fighting for independence from it. Another focuses on the last descendant of the realm's deposed ruling dynasty, who has been exiled to Essos and is plotting a return to the throne, while another story arc follows the Night's Watch, a brotherhood defending the realm against the fierce peoples and legendary creatures of the North.

Game of Thrones attracted a record viewership on HBO and has a broad, active, and international fan base. The series was acclaimed by critics for its acting, complex characters, story, scope, and production values, although its frequent use of nudity and violence (including sexual violence) was criticized; the final season received further criticism for its condensed story and creative decisions, with many considering it a disappointing conclusion. The series received 58 Primetime Emmy Awards, the most by a drama series, including Outstanding Drama Series in 2015, 2016, 2018, and 2019. Its other awards and nominations include three Hugo Awards for Best Dramatic Presentation (2012–2014), a 2011 Peabody Award, and five nominations for the Golden Globe Award for Best Television Series – Drama (2012 and 2015–2018). Many critics and publications have named the show as one of the best television series of all time.
"""

contractions_re = re.compile('(%s)' % '|'.join(contractions.keys()))
def expand_contractions(s, contractions_dict=contractions):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)
sentences = sent_tokenize(passage)    
sentences = [expand_contractions(i) for i in sentences]
sentences = [re.sub('\n', '', i) for i in sentences]

def create_freq_table(text_string):
    stopwords_list = set(stopwords.words('english'))
    
    words = word_tokenize(text_string)
    
    ps = PorterStemmer()
    
    freq_table = {}
    
    for word in words:
        #stem word 
        word = ps.stem(word)
        
        #remove stopwords
        if word in stopwords_list: 
            continue
        elif word in freq_table:
            freq_table[word] += 1
        else:
            freq_table[word] = 1
            
    return freq_table

freq_table = create_freq_table(" ".join(sentences))

def score_sentences(sentences, freq_table):
    
    sentence_value = {}
    
    for sentence in sentences:
        word_count_in_sentence = len(word_tokenize(sentence))
        
        for wordValue in freq_table:
            
            if wordValue.lower() in sentence.lower():                
                if sentence in sentence_value:
                    sentence_value[sentence] += freq_table[wordValue]
                else:
                    sentence_value[sentence] = freq_table[wordValue]

        sentence_value[sentence] = sentence_value[sentence] // word_count_in_sentence
    return sentence_value

def find_average_score(sentence_value):
    sum_values = 0
    
    for entry in sentence_value:
        sum_values += sentence_value[entry]
        
    average = int(sum_values/len(sentence_value))
    
    return average

def generate_summary(sentences, sentence_value, threshold):
    sentence_count = 0
    
    summary = ''
    
    for sentence in sentences:
        if sentence in sentence_value and sentence_value[sentence] > threshold:
            summary += " " + sentence
            sentence_count += 1
            
    return summary

freq_table = create_freq_table(" ".join(sentences))

sentence_scores = score_sentences(sentences, freq_table)

threshold = find_average_score(sentence_scores)

summary = generate_summary(sentences, sentence_scores, 1.0 * threshold)
summary
module_url1 = "https://tfhub.dev/google/universal-sentence-encoder/2"

embed1 = hub.Module(module_url1)

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    message_embeddings = session.run(embed1(sentences))

#generate cosine similarity matrix
sim_matrix = cosine_similarity(message_embeddings)

#create graph and generate scores from pagerank algorithms
nx_graph = nx.from_numpy_array(sim_matrix)
scores = nx.pagerank(nx_graph)

ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
   
num_of_sentences = 3
    
summary = " ".join([i[1] for i in ranked_sentences[:num_of_sentences]])
print(summary)
