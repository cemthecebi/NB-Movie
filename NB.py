from tkinter import *
from tkinter.filedialog import askopenfilename
import tkinter.messagebox

import pandas as pd 
import numpy as np 
from collections import defaultdict
import re
#deneme import



root = Tk(className=" Naive Movie Comment Prediction")
root.geometry("500x300")

class NaiveBayes:
    
    def __init__(self,unique_classes):
	#constructor
        
        self.classes=unique_classes 
        
    def addToBow(self,example,dict_index):
    #fonksiyon gelen example(string) boşluk bazında ayırır ve
	#belirtilen kelimeyi sepete ekler    
        if isinstance(example,np.ndarray): example=example[0]
     
        for token_word in example.split():
        #exampledaki tüm parçaları gezer   
            self.bow_dicts[dict_index][token_word]+=1
            #ve değerini 1 arttırır

    def train(self,dataset,labels):
        #Naive Bayes modelini eğitecek olan ve her kategori ve
	    #sınıf için bir sepet oluşturacak ve hesaplayacak fonksiyon
        self.examples=dataset
        self.labels=labels
        self.bow_dicts=np.array([defaultdict(lambda:0) for index in range(self.classes.shape[0])])

        #Eğer example verisi numpy arr olarak gelmezse onu numpy dizilerine dönüştürür     
        if not isinstance(self.examples,np.ndarray): self.examples=np.array(self.examples)
        if not isinstance(self.labels,np.ndarray): self.labels=np.array(self.labels)

        #her kategori için bir sepet oluşturur	
        for cat_index,cat in enumerate(self.classes):
            #kategorinin tüm örneklerini filtrele
            all_cat_examples=self.examples[self.labels==cat]
            #işlenmiş verileri alır 
            cleaned_examples=[clean_string(cat_example) for cat_example in all_cat_examples]
            cleaned_examples=pd.DataFrame(data=cleaned_examples)
            #kategori için gerekli sepeti oluşturur
            np.apply_along_axis(self.addToBow,1,cleaned_examples,cat_index)   
       
        #Bu aşamada kategoriler için sepetleri oluşturduk ve 
		#şimdi aşağıdaki hesaplamaları yapacağız
		#1- her sınıfın ön olasığı -(prior probability of each class) 
		#2- Kelime haznesi - |V|
		#3- her classın paydasını hesaplanacak [count(c)+|V|+1]

        prob_classes=np.empty(self.classes.shape[0])
        all_words=[]
        cat_word_counts=np.empty(self.classes.shape[0])
        for cat_index,cat in enumerate(self.classes):
            #Her sınıf için prior prob. hesaplar
            prob_classes[cat_index]=np.sum(self.labels==cat)/float(self.labels.shape[0]) 
            #Her sınıf için tüm kelimelerin toplam sayımı hesaplanır
            count=list(self.bow_dicts[cat_index].values())
            cat_word_counts[cat_index]=np.sum(np.array(list(self.bow_dicts[cat_index].values())))+1 # |v| is remaining to be added
            #categorideki tüm kelimeli ekler
            all_words+=self.bow_dicts[cat_index].keys()
                                                     
        
        #her katgorinin içindeki kelimeleri birleştirir |V| tüm
		#kelimelerini elde eder
        self.vocab=np.unique(np.array(all_words))
        self.vocab_length=self.vocab.shape[0]
        #payda değerini hesaplar                          
        denoms=np.array([cat_word_counts[cat_index]+self.vocab_length+1 for cat_index,cat in enumerate(self.classes)])                                                                          
        
        #Bütün verilerimi hesapladı ve bunları ayrı listelerde 
		#tutmamak için (rahat erişim ve bütünlük) her şeyi bir 
		#tuple'ın içine koyacağız. cats_info her elemanı bir değer
		#kümesine sahiptir 0->dict , 1->prior probability, 2-> payda değerini          
        self.cats_info=[(self.bow_dicts[cat_index],prob_classes[cat_index],denoms[cat_index]) for cat_index,cat in enumerate(self.classes)]                               
        self.cats_info=np.array(self.cats_info)                                 
                                                                                    
    def ge_ex_prob(self,test_example):      
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
    #Her test örneiğini tüm sınıflara karşı olasılığını belirler
	#ve sınıf olasılığının maksiumum olanı tahmin eder.          
        predictions=[] #to store prediction of each test example
        for example in test_set: 
            #test örnekler stringleri temizliyoruz.                                  
            cleaned_example=clean_string(example)                            
            post_prob=self.ge_ex_prob(cleaned_example) 
            #en yüksek değeri alıp self e map ediyoruz
            predictions.append(self.classes[np.argmax(post_prob)])
                
        return np.array(predictions) 


def clean_string(str_arg):
    #Bu fonksiyon train ve test verilerini küçük harflere
	#çevirir, birden fazla boşluk varsa onu düzeltir ve
	#özel karakterleri stringden çıkartır.
	
	#özel karakterler stringden çıkarır.
    cleaned_str=re.sub('[^a-z\s]+',' ',str_arg,flags=re.IGNORECASE) 
    #birden fazla space varsa bu tek space dönüştürüldü
    cleaned_str=re.sub('(\s+)',' ',cleaned_str) 
    #temizlenen stringin tümünü küçük harflere çevirir
    cleaned_str=cleaned_str.lower()
  	#temizlenen stringi döndürür  
    return cleaned_str 

""" KAGGLE veriseti yüklenmesi ve sonuçların çıktı olarak alınması"""

def import_train_data():
    cvs_file_path=askopenfilename()
    return cvs_file_path

def import_test_data():
    cvs_file_path=askopenfilename()
    return cvs_file_path


def get_train_test():
    file_path=import_train_data()
    training_set=pd.read_csv(file_path,sep="\t")

    #getting training set examples labels
    y_train=training_set['sentiment'].values
    x_train=training_set['review'].values

    from sklearn.model_selection import train_test_split
    train_data,test_data,train_labels,test_labels=train_test_split(x_train,y_train,shuffle=True,test_size=0.25,random_state=42,stratify=y_train)
    classes=np.unique(train_labels)

    # Eğitim aşaması
    global nb
    nb=NaiveBayes(classes)
    nb.train(train_data,train_labels)

    # Test aşaması
    pclasses=nb.test(test_data)
    test_acc=np.sum(pclasses==test_labels)/float(test_labels.shape[0])
    test_acc_txt = ("Test Set Accuracy: ",test_acc)
    tkinter.messagebox.showinfo("Test Accuracy",test_acc_txt)


def final_predict():
    file_path=import_test_data()
    # kaggle datasetin yüklenmesi
    test=pd.read_csv(file_path,sep='\t')
    Xtest=test.review.values

    #tahminlerin oluşturulması
    pclasses=nb.test(Xtest) 

    #sonuçları csv dosyasına yazarak kaggle üzerinden kontrol için uygun kımsa getiriyoruz.
    kaggle_df=pd.DataFrame(data=np.column_stack([test["id"].values,pclasses]),columns=["id","sentiment"])

    kaggle_df.to_csv("./naive_bayes_model1.csv",index=False)
    tkinter.messagebox.showinfo("SUCCESS",'Predictions Generated and saved to naive_bayes_model.csv')




Button(root,bg="#BEBEBE", text='Browse Labeled Train Data Set', font='Helvetica 12 bold',command=get_train_test).grid(row=1, column=5, padx=120, pady=50)
#Button(root, text='Close',command=root.destroy).grid(row=1, column=1)
Button(root,bg="#BEBEBE", text='Do the Prediction!', font='Helvetica 12 bold',command=final_predict).grid(row=2, column=5)
status = Label(root, text="Processing…", bd=1, relief=SUNKEN, anchor=W)

#uygulama ppenceresi kapanana kadar programın çalışmasını sağlar
root.mainloop()
