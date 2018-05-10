
# coding: utf-8

# In[16]:


from sklearn.feature_extraction import DictVectorizer
def dictvector():
    dict = DictVectorizer(sparse=False)
    x=[{'city':'beijing','temperature':100},{'city':'shanghai','temperature':60},{'city':'shenzhen','temperature':30}]
    data = dict.fit_transform(x)
    print(dict.get_feature_names())
    print(data)
    print(dict.inverse_transform(data))
    print(dict.transform(x))
    return None
if __name__ == '__main__':
    dictvector()


# In[24]:


from sklearn.feature_extraction.text import CountVectorizer
def countvect():
    cv = CountVectorizer()
    X = ["life is is short,i like python","life is long,i dislike python"]
    data = cv.fit_transform(X)
    print(cv.get_feature_names())
    print(data.toarray())
countvect()


# In[31]:


from sklearn.feature_extraction.text import CountVectorizer
import jieba
def cutWord():
    con1 = jieba.cut("今天很残酷，明天更残酷！")
    content1 =list(con1)
    c1 = ' '.join(content1)
    con2 =jieba.cut("生活不止眼前和苟且，还有诗与远方！")
    content2 = list(con2)
    c2 = ' '.join(content2)
    con3 = jieba.cut("心有多大，舞台就有多大！")
    content3 = list(con3)
    c3 = ' '.join(content3)
    return c1,c2,c3
def countvect():
    cv = CountVectorizer()
    c1,c2,c3 = cutWord()
    data = cv.fit_transform([c1,c2,c3])
    print(cv.get_feature_names())
    print(data.toarray())
countvect()


# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
def cutWord():
    con1 = jieba.cut("今天很残酷，明天更残酷！")
    content1 =list(con1)
    c1 = ' '.join(content1)
    con2 =jieba.cut("生活不止眼前和苟且，还有诗与远方！")
    content2 = list(con2)
    c2 = ' '.join(content2)
    con3 = jieba.cut("心有多大，舞台就有多大！")
    content3 = list(con3)
    c3 = ' '.join(content3)
    return c1,c2,c3
def tfidfvect():
    tf = TfidfVectorizer()
    c1,c2,c3 = cutWord()
    data = tf.fit_transform([c1,c2,c3])
    print(tf.get_feature_names())
    print(data.toarray())
tfidfvect()

