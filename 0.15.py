
# coding: utf-8

# In[1]:

### Libraries ###


# In[16]:

## Basic Libraries ##
import sys

import pandas as pd
import re
from nltk.corpus import stopwords
import numpy as np

import matplotlib.pyplot as plt

## Emotional Processing ##
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import re
import numpy as np

## Keyword Extraction ## 
import operator
from rake import *


# In[17]:

### End of Libraries ###


# In[ ]:




# In[18]:

### Global Variables ###


# In[19]:

## Emotional Processing ##
nrc_address = "NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt"
emot_dic = ""

## Keyword Extraction ##
stoppath = "6-1-SmartStoplist.txt"  # SMART stoplist misses some of the lower-scoring keywords in Figure 1.5, which means that the top 1/3 cuts off one of the 4.0 score words in Table 1.1


# In[20]:

### End of Global Variables ###


# In[21]:

### Functions ###


# In[22]:


## Emotional Processing Part 1##

# Read NRC Emotions
def read_nrc_emotions(nrc_address):
    #nrc_address = "NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt"
    data = {}

    with open(nrc_address, "r", encoding="utf-8") as nrc_file:
        for line in nrc_file.readlines():
            splited = line.replace("\n", "").split("\t")
            word, emotion, value = splited[0], splited[1], splited[2]
            if word in data.keys():
                data[word].append((emotion, int(value)))
            else:
                data[word] = [(emotion, int(value))]
                
    return data

## value set for emot_dic ##
emot_dic = read_nrc_emotions(nrc_address)

# Find a word in NRC Emotions
def get_word_emotion(word, emotion):
    emotions = emot_dic[word]
    for emot in emotions:
        if emot[0] == emotion:
            return emot[1]
    return 0.0

# Find a sentence in NRC Emotions
def get_sent_emotion(sent, emotion):
    word_count = len(sent)
    emotion_level = 0.0
    
    if word_count == 0:
        return 0.0
    
    for word in sent.split(" "):
        if word in emot_dic.keys():
            emotion_level = emotion_level + get_word_emotion(word, emotion)
            # print (word, str(emotion_level))
    
    
    emotion_level_avg = emotion_level # / word_count 
    return emotion_level_avg
    
## End of Emotional Processing Part 1 ##


# In[ ]:




# In[23]:

## Emotional Processing Part 2 ##

## Single Emotion
def emotion_story_plot(story,target_emot,color):
    # seperate sentences with "."
    sentences_list = sent_tokenize(story)
    sent_len = len(sentences_list)

    sentences_emot_val =  np.zeros(sent_len)

    for sent,counter in zip(sentences_list,range(0,sent_len-1)):
        # clean sentences
        sent_cl = re.sub(r"[^A-Za-z]", " ", sent)
        target_sent = re.sub(r" +", " ", sent_cl)

        sentences_emot_val[counter] = get_sent_emotion(target_sent,target_emot)

    plt.plot(range(0,sent_len),sentences_emot_val,color)

## Single emotion for a list of stories on single plot
def single_emotion_all_stories_plot(story_list, target_emot,color):
    for story in story_list:
        emotion_story_plot(story,target_emot,color)
    plt.show()

## Draw emotion plot for a list of stories separately
def separate_single_emotion_all_stories_plot(story_list, target_emot,color):
    for story in story_list:
        emotion_story_plot(story,target_emot,color)
        plt.show()

## Draw emotional plot of single story in compare to the rest of the list 
def story_in_context_single_em_plot(story_ind,story_list, target_emot,color_ind,color_all):
    for story in story_list:
        emotion_story_plot(story,target_emot,color_all)
    
    emotion_story_plot(story_ind,target_emot,color_ind)
    plt.show()

        
## Draw Plots of Multiple Emotion in a single story
def multi_emotions_single_story(story, emot_lst, color_lst):
    for emotion, color in zip(emot_lst, color_lst):
        emotion_story_plot(story,emotion,color)
    plt.show()

## Draw Emotional Plot of a single story based on a weighted average of a list of emotion
def weighted_emotions_single_story(story,emot_lst,weight_lst,color):
    # seperate sentences with "."
    sentences_list = sent_tokenize(story)
    sent_len = len(sentences_list)
    total_weight = sum(weight_lst)
    
    sentences_emot_val =  np.zeros(sent_len)

    for emotion,weight in zip(emot_lst,weight_lst):
        for sent,counter in zip(sentences_list,range(0,sent_len-1)):
            # clean sentences
            sent_cl = re.sub(r"[^A-Za-z]", " ", sent)
            target_sent = re.sub(r" +", " ", sent_cl)

            sentences_emot_val[counter] += weight * (get_sent_emotion(target_sent,emotion))

    sentences_weighted_em = [(x) for x in sentences_emot_val]
    plt.plot(range(0,sent_len),sentences_weighted_em,color)

## Draw weighted emotional plot of a list of stories
def weighted_emotions_all_stories_plot(story_list, emot_lst, weight_lst, color):
    for story in story_list:
        weighted_emotions_single_story(story,emot_lst,weight_lst, color)
    plt.show()
    
## Draw weighted emotional plot of a list of stories, speratedly
def separated_weighted_emotions_all_stories_plot(story_list, emot_lst, weight_lst, color):
    for story in story_list:
        weighted_emotions_single_story(story,emot_lst,weight_lst, color)
        plt.show()

## Draw weighted emotional plot of single story in compare to the rest of the list 
def story_in_context_single_em_plot(story_ind, story_list, emot_lst, weight_lst, color_ind, color_all):
    for story in story_list:
        weighted_emotions_single_story(story, emot_lst, weight_lst, color_all)
    
    weighted_emotions_single_story(story_ind, emot_lst, weight_lst, color_ind)
    plt.show()
    
def emotional_plot_of_story_lst(story_lst, emotion_lst, weight_lst):
    story_lst_len = len (story_lst)
    sentences_emot_val =  np.zeros(story_lst_len)
    for story,counter in zip(story_lst,range(0,len(story_lst)-1)):
        sentences_emot_val[counter] = emotional_degree_of_story(story, emotion_lst, weight_lst)
    plt.plot(range(0,story_lst_len),sentences_emot_val,'bo')
    plt.show()
    
def emotional_plot_of_story_in_a_lst(story_indiv_index, story_lst, emotion_lst, weight_lst):
    story_lst_len = len (story_lst)
    sentences_emot_val =  np.zeros(story_lst_len)
    for story,counter in zip(story_lst,range(0,len(story_lst)-1)):
        sentences_emot_val[counter] = emotional_degree_of_story(story, emotion_lst, weight_lst)
    plt.plot(range(0,story_lst_len),sentences_emot_val,'bo')
    plt.plot(story_indiv_index,emotional_degree_of_story(story_lst[story_indiv_index], emotion_lst, weight_lst),'ro')
    plt.show()
    

### Numerical ####
## Length of sentences ##
def avg_length_of_sent(story):
    # seperate sentences with "."
    sentences_list = sent_tokenize(story)
    sent_len = len(sentences_list)
    
    avg_sentences_len = 0.0

    for sent,counter in zip(sentences_list,range(0,sent_len-1)):
        # clean sentences
        sent_cl = re.sub(r"[^A-Za-z]", " ", sent)
        target_sent = re.sub(r" +", " ", sent_cl)
        
        avg_sentences_len += len(target_sent.split(" "))
    
    if sent_len != 0:
        avg_sentences_len /= sent_len
        return avg_sentences_len
    
    return avg_sentences_len

def avg_length_of_story_in_lst(story_lst):
    
    story_len = len(story_lst)
    
    avg_story_sent_len = 0.0

    for story in story_lst:
        avg_story_sent_len += avg_length_of_sent(story)
    
    if story_len != 0:
        avg_story_sent_len /= story_len
        return avg_story_sent_len
    return avg_story_sent_len
    
## Range of Emotion
def range_of_emotion(story,emot_lst,weight_lst):
    # seperate sentences with "."
    sentences_list = sent_tokenize(story)
    sent_len = len(sentences_list)
    #total_weight = sum(weight_lst)
    
    sentences_emot_val =  np.zeros(sent_len)

    for emotion,weight in zip(emot_lst,weight_lst):
        for sent,counter in zip(sentences_list,range(0,sent_len-1)):
            # clean sentences
            sent_cl = re.sub(r"[^A-Za-z]", " ", sent)
            target_sent = re.sub(r" +", " ", sent_cl)

            sentences_emot_val[counter] += weight * (get_sent_emotion(target_sent,emotion))

    #sentences_weighted_em = [(x) for x in sentences_emot_val]
    min_em = min(sentences_emot_val)
    max_em = max(sentences_emot_val)
    
    return abs(max_em-min_em)

# average of emotional range in a collection of story
def avg_range_of_emotion_in_all_story(story_lst,emot_lst,weight_lst):
    avg_emotion_range = 0.0
    counter = 0
    
    for story in story_lst:
        avg_emotion_range += range_of_emotion(story,emot_lst,weight_lst)
        counter += 1
    
    return (avg_emotion_range/counter)

# highest emotional range in a collection of story
def max_range_of_emotion_in_all_story(story_lst,emot_lst,weight_lst):
    max_emotion_range = 0.0
    
    for story in story_lst:
        r_of_e = range_of_emotion(story,emot_lst,weight_lst)
        max_emotion_range = r_of_e if max_emotion_range < r_of_e else max_emotion_range 
    
    return max_emotion_range


## Emotional Mutation
def mutation_of_emotion(story,emot_lst,weight_lst):
    # seperate sentences with "."
    sentences_list = sent_tokenize(story)
    sent_len = len(sentences_list)
    #total_weight = sum(weight_lst)
    
    sentences_emot_val =  np.zeros(sent_len)

    for emotion,weight in zip(emot_lst,weight_lst):
        for sent,counter in zip(sentences_list,range(0,sent_len-1)):
            # clean sentences
            sent_cl = re.sub(r"[^A-Za-z]", " ", sent)
            target_sent = re.sub(r" +", " ", sent_cl)

            sentences_emot_val[counter] += weight * (get_sent_emotion(target_sent,emotion))

    mutation_val = 0.0
    
    em_diff = [abs(sentences_emot_val[i]-sentences_emot_val[i-1]) for i in range(1,len(sentences_emot_val))]
    mutation_val = sum(em_diff)
    
    return abs(mutation_val)

def avg_mutation_of_emotion_in_all_story(story_lst,emot_lst,weight_lst):
    avg_emotion_mutation = 0.0
    counter = 0
    
    for story in story_lst:
        avg_emotion_mutation += mutation_of_emotion(story,emot_lst,weight_lst)
        counter += 1
    
    return (avg_emotion_mutation/counter)

def max_mutation_of_emotion_in_all_story(story_lst,emot_lst,weight_lst):
    max_emotion_mutation = 0.0
    
    for story in story_lst:
        m_of_e = mutation_of_emotion(story,emot_lst,weight_lst)
        max_emotion_mutation = m_of_e if max_emotion_mutation < m_of_e else max_emotion_mutation 
    
    return max_emotion_mutation

## Change point
    
    
## Emotional Level

# single sentence #
def emotional_level_sent(sent, emot_lst, weight_lst):
    level = 0.0
    for emot, wt in zip(emot_lst, weight_lst):
        level += (wt * get_sent_emotion(sent,emot))
    
    return level
    
# max sentence in a story
def max_emot_sent(story,emot_lst,weight_lst):
    max_em_level = 0.0
    max_em_sent = ""
    
    # seperate sentences with "."
    sentences_list = sent_tokenize(story)
    sent_len = len(sentences_list)

    #sentences_emot_val =  np.zeros(sent_len)

    for sent,counter in zip(sentences_list,range(0,sent_len-1)):
        # clean sentences
        sent_cl = re.sub(r"[^A-Za-z]", " ", sent)
        target_sent = re.sub(r" +", " ", sent_cl)
        
        em_level = emotional_level_sent(sent, emot_lst, weight_lst)
        
        if em_level > max_em_level:
            max_em_level = em_level
            max_em_sent = sent
            
    return (max_em_sent,max_em_level)

# emotional summery of a collection of stories
def emotional_summery(story_lst,emot_lst,weight_lst):
    emotional_sents = []
    for story in story_lst:
        emotional_sent = max_emot_sent(story,['joy','sadness'],[1.0,1.0])
        emotional_sents.append(emotional_sent[0])
    
    emotional_summary = ' '.join(emotional_sents)
    return emotional_summary

# remove emotional words
def remove_emot_words(text, emot_lst):
    emotless_txt = ""
    for word in text.split(" "):
        flag = True
        if word in emot_dic.keys():
            for emot in emot_lst:
                if  get_word_emotion(word, emot) != 0.0:
                    flag = False
                    break
        if flag:
            emotless_txt += word
            emotless_txt += " "
    return emotless_txt
                
def emotional_degree_of_story(story, emotion_lst, weight_lst):
    # seperate sentences with "."
    sentences_list = sent_tokenize(story)
    sent_len = len(sentences_list)
    total_weight = sum(weight_lst)
    
    sentences_emot_val =  0.0

    for emotion,weight in zip(emotion_lst,weight_lst):
        for sent,counter in zip(sentences_list,range(0,sent_len-1)):
            # clean sentences
            sent_cl = re.sub(r"[^A-Za-z]", " ", sent)
            target_sent = re.sub(r" +", " ", sent_cl)

            sentences_emot_val += weight * (get_sent_emotion(target_sent,emotion))

    return sentences_emot_val

## End of Emotional Processing Part 2 ##


# In[24]:

## Keywords Extraction ##

# Single Text #
def keyword_extraction_from_text(text):
    rake_object = Rake(stoppath, min_char_length=1, max_words_length=2, min_keyword_frequency=1)
    keywords = rake_object.run(text)
    return keywords

# List of stories #
def keywords_in_story_lst(story_lst):
    joint_txt = ' '.join(story_lst)
    keywords = keyword_extraction_from_text(joint_txt)
    return keywords

## End of Keywords Extraction ##


# In[25]:

### End of Functions ###


# In[26]:

# Reading story data
feeding_stories = pd.read_csv('data/feedingamerica_rep.csv', encoding='latin1')

Supported_df = feeding_stories[feeding_stories['MyTag']=='Supported']
Supporter_df = feeding_stories[feeding_stories['MyTag']=='Supporter']

#normalize!
Supported_df = Supported_df[Supported_df.row_id < 72]

print("Suppoters number of row = ", Supporter_df.shape[0], "\nSupported number of rows = ", Supported_df.shape[0])


# In[30]:

# Main Testings Part#

###### PLOTS #######
# story in context #
## story_in_context_single_em_plot(feeding_stories['story_cl'][0], feeding_stories['story_cl'], ['joy'], [1.0],'b','r')
story_in_context_single_em_plot(feeding_stories['story_cl'][0], Supported_df['story_cl'], ['sadness'], [1.0],'b','r')
story_in_context_single_em_plot(feeding_stories['story_cl'][0], Supporter_df['story_cl'], ['sadness'], [1.0],'b','r')

# single emotions in supporters and supporteds #
single_emotion_all_stories_plot(Supported_df['story_cl'], 'joy','r')
single_emotion_all_stories_plot(Supporter_df['story_cl'], 'joy','b')

# single emotions in supporters and supporteds #
single_emotion_all_stories_plot(Supported_df['story_cl'], 'sadness','r')
single_emotion_all_stories_plot(Supporter_df['story_cl'], 'sadness','b')

# single emotions in supporters and supporteds #
single_emotion_all_stories_plot(Supported_df['story_cl'], 'surprise','r')
single_emotion_all_stories_plot(Supporter_df['story_cl'], 'surprise','b')

# single emotions in supporters and supporteds #
single_emotion_all_stories_plot(Supported_df['story_cl'], 'fear','r')
single_emotion_all_stories_plot(Supporter_df['story_cl'], 'fear','b')

# single emotions in supporters and supporteds #
single_emotion_all_stories_plot(Supported_df['story_cl'], 'anger','r')
single_emotion_all_stories_plot(Supporter_df['story_cl'], 'anger','b')

# single emotions in supporters and supporteds #
single_emotion_all_stories_plot(Supported_df['story_cl'], 'disgust','r')
single_emotion_all_stories_plot(Supporter_df['story_cl'], 'disgust','b')

# single emotions in supporters and supporteds #
single_emotion_all_stories_plot(Supported_df['story_cl'], 'trust','r')
single_emotion_all_stories_plot(Supporter_df['story_cl'], 'trust','b')

# single emotions in supporters and supporteds #
single_emotion_all_stories_plot(Supported_df['story_cl'], 'positive','r')
single_emotion_all_stories_plot(Supporter_df['story_cl'], 'positive','b')

# single emotions in supporters and supporteds #
single_emotion_all_stories_plot(Supported_df['story_cl'], 'negative','r')
single_emotion_all_stories_plot(Supporter_df['story_cl'], 'negative','b')

weighted_emotions_all_stories_plot(Supported_df['story_cl'], ['positive','negative'], [1.0,-1.0],'r')
weighted_emotions_all_stories_plot(Supporter_df['story_cl'], ['positive','negative'], [1.0,-1.0], 'b')

emotional_plot_of_story_lst(Supported_df['story_cl'], ['positive','negative'], [1.0,-1.0])
emotional_plot_of_story_lst(Supporter_df['story_cl'], ['positive','negative'], [1.0,-1.0])
emotional_plot_of_story_in_a_lst(0, feeding_stories['story_cl'], ['positive','negative'], [1.0,-1.0])
###### Numerics #######
## Emotion = "Positive - Negative" ##
supported_avg_range = avg_range_of_emotion_in_all_story(Supported_df['story_cl'],['positive','negative'],[1.0,-1.0])
supporter_avg_range = avg_range_of_emotion_in_all_story(Supporter_df['story_cl'],['positive','negative'],[1.0,-1.0])
print("Supported Avg range of emotions = ", supported_avg_range, "\nSupporter Avg range of emotions = ",supporter_avg_range)

supported_max_range = max_range_of_emotion_in_all_story(Supported_df['story_cl'],['positive','negative'],[1.0,-1.0])
supporter_max_range = max_range_of_emotion_in_all_story(Supporter_df['story_cl'],['positive','negative'],[1.0,-1.0])
print("Supported Max range of emotions = ", supported_max_range, "\nSupporter Max range of emotions = ",supporter_max_range)

supported_avg_mut = avg_mutation_of_emotion_in_all_story(Supported_df['story_cl'],['positive','negative'],[1.0,-1.0])
supporter_avg_mut = avg_mutation_of_emotion_in_all_story(Supporter_df['story_cl'],['positive','negative'],[1.0,-1.0])
print("Supported Avg mutation of emotions = ", supported_avg_mut, "\nSupporter Avg mutation of emotions = ",supporter_avg_mut)

supported_max_mut = max_mutation_of_emotion_in_all_story(Supported_df['story_cl'],['positive','negative'],[1.0,-1.0])
supporter_max_mut = max_mutation_of_emotion_in_all_story(Supporter_df['story_cl'],['positive','negative'],[1.0,-1.0])
print("Supported Max mutation of emotions = ", supported_max_mut, "\nSupporter Max mutation of emotions = ",supporter_max_mut)

## Emotion = "Joy - Sadness" ##
supported_avg_range = avg_range_of_emotion_in_all_story(Supported_df['story_cl'],['joy','sadness'],[1.0,-1.0])
supporter_avg_range = avg_range_of_emotion_in_all_story(Supporter_df['story_cl'],['joy','sadness'],[1.0,-1.0])
print("Supported Avg range of emotions = ", supported_avg_range, "\nSupporter Avg range of emotions = ",supporter_avg_range)

supported_max_range = max_range_of_emotion_in_all_story(Supported_df['story_cl'],['joy','sadness'],[1.0,-1.0])
supporter_max_range = max_range_of_emotion_in_all_story(Supporter_df['story_cl'],['joy','sadness'],[1.0,-1.0])
print("Supported Max range of emotions = ", supported_max_range, "\nSupporter Max range of emotions = ",supporter_max_range)

supported_avg_mut = avg_mutation_of_emotion_in_all_story(Supported_df['story_cl'],['joy','sadness'],[1.0,-1.0])
supporter_avg_mut = avg_mutation_of_emotion_in_all_story(Supporter_df['story_cl'],['joy','sadness'],[1.0,-1.0])
print("Supported Avg mutation of emotions = ", supported_avg_mut, "\nSupporter Avg mutation of emotions = ",supporter_avg_mut)

supported_max_mut = max_mutation_of_emotion_in_all_story(Supported_df['story_cl'],['joy','sadness'],[1.0,-1.0])
supporter_max_mut = max_mutation_of_emotion_in_all_story(Supporter_df['story_cl'],['joy','sadness'],[1.0,-1.0])
print("Supported Max mutation of emotions = ", supported_max_mut, "\nSupporter Max mutation of emotions = ",supporter_max_mut)


###### Content #######
supported_em_sum = emotional_summery(Supported_df['story_cl'],['positive','negative'],[1.0,1.0])
print("Supported Emotional Summery:\n",supported_em_sum)
supported_emoless_sum = remove_emot_words(supported_em_sum, ['positive','negative'])

print("Supported Raw Keywords:\n",keywords_in_story_lst(Supported_df['story_cl'])[0:50])
print("Supported Emotional Summery:\n",supported_em_sum)
print("Supported Emotional Keywords:\n", keyword_extraction_from_text(supported_em_sum )[0:50])
print ("Supported Emotionless Keywords:\n", keyword_extraction_from_text(supported_emoless_sum)[0:50])

supporter_em_sum = emotional_summery(Supporter_df['story_cl'],['positive','negative'],[1.0,1.0])
supporter_emoless_sum = remove_emot_words(supporter_em_sum, ['positive','negative'])

print("Supporter Raw Keywords:\n",keywords_in_story_lst(Supporter_df['story_cl'])[0:30])
print("Supporter Emotional Summery:\n",supporter_em_sum)
print("Supporter Emotional Keywords:\n", keyword_extraction_from_text(supporter_em_sum )[0:30])
print ("Supporter Emotionless Keywords:\n", keyword_extraction_from_text(supporter_emoless_sum)[0:30])

print ("Average Length of sentence length for Supported narratives =", avg_length_of_story_in_lst(Supported_df['story_cl']))
print ("Average Length of sentence length for Supporter narratives =", avg_length_of_story_in_lst(Supporter_df['story_cl']))



# In[ ]:




# In[ ]:




# In[ ]:



