from tkinter import *
from tkinter.filedialog import askopenfilename
import tkinter.messagebox

import pandas as pd 
import numpy as np 
from collections import defaultdict
import re




root = Tk(className=" Naive Movie Comment Prediction")
root.geometry("500x300")

def preprocess_string(str_arg):
    cleaned_str=re.sub('[^a-z\s]+',' ',str_arg,flags=re.IGNORECASE) 
    cleaned_str=re.sub('(\s+)',' ',cleaned_str) 
    cleaned_str=cleaned_str.lower()
    
    return cleaned_str 

class NaiveBayes:
    
    def __init__(self,unique_classes):
        
        self.classes=unique_classes 
        
    def addToBow(self,example,dict_index):
        if isinstance(example,np.ndarray): example=example[0]
     
        for token_word in example.split():
          
            self.bow_dicts[dict_index][token_word]+=1
            
    def train(self,dataset,labels):
        self.examples=dataset
        self.labels=labels
        self.bow_dicts=np.array([defaultdict(lambda:0) for index in range(self.classes.shape[0])])
               
        if not isinstance(self.examples,np.ndarray): self.examples=np.array(self.examples)
        if not isinstance(self.labels,np.ndarray): self.labels=np.array(self.labels)
            
        for cat_index,cat in enumerate(self.classes):
          
            all_cat_examples=self.examples[self.labels==cat] 
            cleaned_examples=[preprocess_string(cat_example) for cat_example in all_cat_examples]
            cleaned_examples=pd.DataFrame(data=cleaned_examples)
            np.apply_along_axis(self.addToBow,1,cleaned_examples,cat_index)   
       
        prob_classes=np.empty(self.classes.shape[0])
        all_words=[]
        cat_word_counts=np.empty(self.classes.shape[0])
        for cat_index,cat in enumerate(self.classes):
           
            prob_classes[cat_index]=np.sum(self.labels==cat)/float(self.labels.shape[0]) 
            
            count=list(self.bow_dicts[cat_index].values())
            cat_word_counts[cat_index]=np.sum(np.array(list(self.bow_dicts[cat_index].values())))+1 # |v| is remaining to be added
            
            all_words+=self.bow_dicts[cat_index].keys()
                                                     
        
        
        self.vocab=np.unique(np.array(all_words))
        self.vocab_length=self.vocab.shape[0]
                                  
        denoms=np.array([cat_word_counts[cat_index]+self.vocab_length+1 for cat_index,cat in enumerate(self.classes)])                                                                          
                  
        self.cats_info=[(self.bow_dicts[cat_index],prob_classes[cat_index],denoms[cat_index]) for cat_index,cat in enumerate(self.classes)]                               
        self.cats_info=np.array(self.cats_info)                                 
                                                                                    
    def getExampleProb(self,test_example):      
        likelihood_prob=np.zeros(self.classes.shape[0]) 
        
        for cat_index,cat in enumerate(self.classes): 
                             
            for test_token in test_example.split(): 
                                     
                test_token_counts=self.cats_info[cat_index][0].get(test_token,0)+1
                
                test_token_prob=test_token_counts/float(self.cats_info[cat_index][2])                              
                
                likelihood_prob[cat_index]+=np.log(test_token_prob)
                                              
        post_prob=np.empty(self.classes.shape[0])
        for cat_index,cat in enumerate(self.classes):
            post_prob[cat_index]=likelihood_prob[cat_index]+np.log(self.cats_info[cat_index][1])                                  
      
        return post_prob
    
    def test(self,test_set):       
        predictions=[] #to store prediction of each test example
        for example in test_set: 
                                              
            #preprocess the test example the same way we did for training set exampels                                  
            cleaned_example=preprocess_string(example) 
             
            #simply get the posterior probability of every example                                  
            post_prob=self.getExampleProb(cleaned_example) #get prob of this example for both classes
            
            #simply pick the max value and map against self.classes!
            predictions.append(self.classes[np.argmax(post_prob)])
                
        return np.array(predictions) 

""" KAGGLE"""

def import_train_data():
    cvs_file_path=askopenfilename()
    return cvs_file_path

def import_test_data():
    cvs_file_path=askopenfilename()
    return cvs_file_path


def addTrainTest():
    ##TO-DO bu kısımı tkiner üzerinden aktif edeceksin.
    #training_set=pd.read_csv('./data/labeledTrainData.tsv',sep='\t') # reading the training data-set
    file_path=import_train_data()
    training_set=pd.read_csv(file_path,sep="\t")

    #getting training set examples labels
    y_train=training_set['sentiment'].values
    x_train=training_set['review'].values

    """
        Again - it's not a problem at all if you didnt understand this block of code - You should just know that some
        train & test data is being loaded and saved in their corresponding variables

    """

    from sklearn.model_selection import train_test_split
    train_data,test_data,train_labels,test_labels=train_test_split(x_train,y_train,shuffle=True,test_size=0.25,random_state=42,stratify=y_train)
    classes=np.unique(train_labels)

    # Training phase....
    global nb
    nb=NaiveBayes(classes)
    nb.train(train_data,train_labels)

    # Testing phase 

    pclasses=nb.test(test_data)
    test_acc=np.sum(pclasses==test_labels)/float(test_labels.shape[0])

    #print ("Test Set Accuracy: ",test_acc) # Output : Test Set Accuracy:  0.84224 :)
    tkinter.messagebox.showinfo("Test Accuracy",test_acc)


def finalPrediction():
    file_path=import_test_data()
    # Loading the kaggle test dataset
    test=pd.read_csv(file_path,sep='\t')
    Xtest=test.review.values

    #generating predictions....
    pclasses=nb.test(Xtest) 

    #writing results to csv to uplaoding on kaggle!
    kaggle_df=pd.DataFrame(data=np.column_stack([test["id"].values,pclasses]),columns=["id","sentiment"])

    #TO-DO bu kısımda kullanıcı için browse kısmı açılacak ve kullancıı dosyaı kaydetmek istediği lokasyonu seçecek.
    kaggle_df.to_csv("./naive_bayes_model1.csv",index=False)
    #print ('Predcitions Generated and saved to naive_bayes_model.csv')
    tkinter.messagebox.showinfo("SUCCESS",'Predictions Generated and saved to naive_bayes_model.csv')



Label(root,text="File Path").grid(row=0,column=0)

Button(root, text='Browse Labeled Train Data Set',command=addTrainTest).grid(row=1, column=1)
#Button(root, text='Close',command=root.destroy).grid(row=1, column=1)
Button(root, text='Do the Prediction!',command=finalPrediction).grid(row=2, column=1)





root.mainloop()
