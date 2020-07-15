import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import json

#The file must not contain any comma each row. if there is a comma in between text, it will throw an error.
#remove commas from text part to avoid errors.
def getData():
    df = pd.read_csv('plagcheckfile.csv')
    trainingDictionary = dict()
    listTexts = df['Text'].values.tolist()
    finallist=[]        
    finalDictionary = {'intents':'demo'}
    print(listTexts)
    for i in range (0,4):
        textDictionary = {"tag":i}
        listTexts_i = cleanText(listTexts[i])
        print(listTexts_i)
        textDictionary.update(texts = listTexts_i)
        finallist.append(textDictionary)
    finalDictionary={"intents":finallist}       
    return finalDictionary

def cleanText(Text):
    # Remove new lines within message
    cleanedText = Text.replace('\n',' ').lower()
    # Deal with some weird tokens
    cleanedText = cleanedText.replace("\xc2\xa0", "")
    # Remove punctuation
    cleanedText = re.sub('([,])','', cleanedText)
    # Remove multiple spaces in message
    cleanedText = re.sub(' +',' ', cleanedText)
    cleanedText = cleanedText.encode('ascii', 'ignore').decode('ascii')
    return cleanedText

combinedDictionary = dict()
print ('Getting Text Data')
combinedDictionary.update(getData())
print ('Total len of dictionary', len(combinedDictionary))

print ('Saving text data dictionary')
np.save('textDictionary.npy', combinedDictionary)

with open('file.txt', 'w') as file:
     file.write(json.dumps(combinedDictionary))
