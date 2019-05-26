#Import all the dependencies
import gensim
from gensim.models.doc2vec import Doc2Vec  ,TaggedLineDocument
from os import listdir
from os.path import isfile, join
from datasets import DATASET
import pickle
import numpy as np
import csv
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
import json
epochs = 8

def trainsrcfiles(documents):
    model1 = gensim.models.Doc2Vec(documents,vector_size=10, window=5, min_count=1, workers=4 , dm=1)
    # start training
    for epoch in range(epochs):
        #print ('Now training epoch %s'%epoch)
        model1.train(documents,total_examples=model1.corpus_count,epochs=epochs)
        model1.alpha -= 0.002  # decrease the learning rate
        model1.min_alpha = model1.alpha  # fix the learning rate, no decay
    return model1.wv

def bugreports(documents):
    
    model2 = gensim.models.Doc2Vec(documents, vector_size=10, window=2, min_count=1, workers=4 , dm = 1)
    # start training
    for epoch in range(epochs):
        #print ('Now training epoch %s'%epoch)
        model2.train(documents,total_examples=model2.corpus_count,epochs=epochs)
        model2.alpha -= 0.002  # decrease the learning rate
        model2.min_alpha = model2.alpha  # fix the learning rate, no decay
    model2.save('repvec.model')
    return  model2.wv

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def main():
    with open(DATASET.root / 'preprocessed_src.pickle', 'rb') as file:
        src_files = pickle.load(file)
    with open(DATASET.root / 'preprocessed_reports.pickle', 'rb') as file:
        bug_reports = pickle.load(file)
    Sr= {'file_name':  [src.file_name['unstemmed']
                    for src in src_files.values()]  , 
        'class_name': [src.class_names['unstemmed']
                    for src in src_files.values()] , 
         'method_name':[src.method_names['unstemmed']
                    for src in src_files.values()] ,
        'comments': [src.comments['unstemmed']
                    for src in src_files.values()],
        'attributes': [src.attributes['unstemmed']
                    for src in src_files.values()]}

    pr= {'Report_summary':[report.summary['unstemmed']
                    for report in bug_reports.values()]  , 
        'Report_description':[report.description['unstemmed']
                    for report in bug_reports.values()]}

    df = DataFrame(Sr, columns= ['file_name', 'class_name' , 'method_name' , 'comments' , 'attributes'  ])
    rdf = DataFrame(pr, columns= ['Report_summary', 'Report_description'])
    df.to_csv (r'E:\ICS\year three\Semester Two\GP\final\last\trail\finall\src.csv', index = None, header=True) 
    rdf.to_csv (r'E:\ICS\year three\Semester Two\GP\final\last\trail\finall\bugreps.csv', index = None, header=True)

    srcco = pd.read_csv("src.csv")
    reps = pd.read_csv("bugreps.csv")
    fileNames = {}

    srckeyvectors =[]
    for index, row in srcco.iterrows():
        
        label = str(index)
        fileNames[index] = row[0]
        src = trainsrcfiles([gensim.models.doc2vec.TaggedDocument(words=row, tags=[label ])])
        srckeyvectors.append(src.vectors)
        
    reportkeyvectors =[]
  
    for index, row in reps.iterrows():
        label = str(index)
        reports = trainsrcfiles([gensim.models.doc2vec.TaggedDocument(words=row, tags=[label ])])
        reportkeyvectors.append(reports.vectors)

    reportDict = {}
    for index, reportVector in enumerate(reportkeyvectors):
        for  srcVector in enumerate(srckeyvectors):

            cosineSimilarity =  cosine_similarity(srcVector[1], reportVector)
            cosineSimilarity = np.mean(cosineSimilarity)

            try:
                reportDict[index].append(cosineSimilarity)
            except:
                reportDict[index] = [cosineSimilarity]

    allSimis = []
    for i in range(0,len(reportDict)):
        allSimis.append( reportDict.get(i))

    
    dumped = json.dumps(allSimis,cls=NumpyEncoder)
    with open('similarities4.json', 'w') as file:
        json.dump(dumped , file)

if __name__ == '__main__':
    main()