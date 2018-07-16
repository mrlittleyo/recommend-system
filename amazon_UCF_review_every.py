#encoding=utf-8
import nltk
import pandas as pd
from lda import lda
from nltk.corpus import stopwords, brown
import numpy as np
from nltk.stem import WordNetLemmatizer
from math import *


def loadData():
    trainSet = {}
    trainprodUser = {}
    testSet = {}
    testprodUser = {}
    u2u = {}
    #读取数据
    traindata = pd.read_csv('D:\\bangsun\data\\train.csv')
    testdata = pd.read_csv('D:\\bangsun\data\\test.csv')
    #加载训练集
    userlist = list(traindata['REVIEWER_ID'])
    itemlist = list(traindata['ASIN'])
    helpful = list(traindata['HELPFUL'])
    helpless = list(traindata['HELPLESS'])
    review_text = list(traindata['REVIEW_TEXT'])
    summary = list(traindata['SUMMARY'])
    ratelist = list(traindata['OVERALL'])
    timelist = list(traindata['UNIX_REVIEW_TIME'])
    m = len(userlist)
    for i in range(m):
        userId = userlist[i]
        itemId = itemlist[i]
        review = review_text[i]
        rating = ratelist[i]
        public_info = helpful[i]-helpless[i]
        timestamp = timelist[i]
        trainSet.setdefault(userId, {})
        trainSet[userId].setdefault(itemId, [review,rating,public_info])
        #trainSet记录每个用户对每个电影的评分，形如：{'userid':{'itemid':[review,rating,public_info]}}
        trainprodUser.setdefault(itemId, [])
        trainprodUser[itemId].append(userId.strip())
    # print "trainSet:", trainSet.items()[0]
    # print "trainprodUser", trainprodUser.items()[0]
    # print "训练集用户数：", len(trainSet.keys())
    # print "训练集电影数：", len(trainprodUser.keys())
    # 加载测试集
    userlist = list(testdata['REVIEWER_ID'])
    itemlist = list(testdata['ASIN'])
    helpful = list(testdata['HELPFUL'])
    helpless = list(testdata['HELPLESS'])
    review_text = list(testdata['REVIEW_TEXT'])
    summary = list(testdata['SUMMARY'])
    ratelist = list(testdata['OVERALL'])
    timelist = list(testdata['UNIX_REVIEW_TIME'])
    m = len(userlist)
    for i in range(m):
        userId = userlist[i]
        itemId = itemlist[i]
        review = review_text[i]
        rating = ratelist[i]
        public_info = helpful[i] - helpless[i]
        timestamp = timelist[i]
        testSet.setdefault(userId, {})
        testSet[userId].setdefault(itemId, [review,rating,public_info])
        testprodUser.setdefault(itemId, [])
        testprodUser[itemId].append(userId.strip())
    # print "测试集用户数：",len(testSet.keys())
    # print '测试集商品数', len(testprodUser.keys())
    # print "testSet:",testSet.items()[0]

    # 生成用户共有电影矩阵
    for m in trainprodUser.keys(): #m=itemid
        for u in trainprodUser[m]: #trainprodUser[m]为用户列表，u=userid
            u2u.setdefault(u, {})
            for n in trainprodUser[m]:
                if u != n:
                    u2u[u].setdefault(n, [])
                    u2u[u][n].append(m)
    # print "训练集u2u:",u2u.items()[0]
    # print "训练集有公共电影的用户数：",len(u2u.keys())
    return trainSet, testSet, u2u

def read_stopwords():
    stop_words = pd.read_csv('D:\\bangsun\stop\__MACOSX\uwm-workshop\data\._stoplist.csv',header=-1)
    #print stop_words.columns
    stop_list = list(stop_words[0])
    stop_list.append("n't")
    return set(stop_list)

def review_process(trainSet):
    stop_words = read_stopwords()
    NLTK_character = {}
    NLTK_character_accuracy = []
    user_reviews = {}
    num=0
    for user in trainSet.keys():
        # num+=1
        # if num==11:break

        for item in trainSet[user]:
            text = trainSet[user][item][0]
            rating = trainSet[user][item][1]
            public_info = trainSet[user][item][2]
            #print text,rating,public_info
            reviews=str(text).strip()
            # 大小写转换
            reviews = reviews.replace('"', '\'').lower().replace('&#34',' ').replace('.',' ').replace('/',' ')
            #print reviews
            #引入停用词
            reviews_text = nltk.word_tokenize(reviews)
            english_punctuations = [',', '.', ':', ';', '?', '&#34','(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','/']
            reviews_text = [word for word in reviews_text if word not in english_punctuations]
            reviews_set = [word for word in reviews_text if word not in stop_words]
            small_length = []
            for word in  reviews_set:
                if len(word)<=2 or "'" in word:
                    small_length.append(word)
            reviews_set = [word for word in reviews_set if word not in small_length]
            # 词形还原
            wnl = WordNetLemmatizer()
            reviews_origin = []
            for word in reviews_set:
                reviews_origin.append(wnl.lemmatize(word))
            # 再次引入停用词
            stops = set(stopwords.words("english"))
            reviews_set = [word for word in reviews_origin if word not in stops]
            #词性标注
            brown_tagged_sents = brown.tagged_sents(categories='news')  #读取brown语料库
            size = int(len(brown_tagged_sents) * 0.8)  #将语料库的前80%作为词性标注器的参照，用另外20%作为对词性标注器的验证
            train_sents = brown_tagged_sents[:size]
            test_sents = brown_tagged_sents[size:]
            #组合标注器
            t0 = nltk.DefaultTagger('NN')
            t1 = nltk.UnigramTagger(train_sents, backoff=t0)
            t2 = nltk.BigramTagger(train_sents, backoff=t1)
            reviews_tag_list = t2.tag(reviews_set)
            #print reviews_tag_list
            accuracy = t2.evaluate(test_sents)
            NLTK_character_accuracy.append(accuracy)
            review_filter_list = []
            for i in reviews_tag_list:
                character = i[-1]
                NLTK_character[character] = NLTK_character.setdefault(character,0)+1
                # if character =='NN' or character =='JJ':
                #     review_filter_list.append(i)
            user_reviews.setdefault(user,{})
            user_reviews[user].setdefault(item,[reviews_tag_list,rating,public_info])
    pd.DataFrame(NLTK_character).to_csv('D:\\bangsun\data\\nltk_character.csv')
    return user_reviews,NLTK_character,NLTK_character_accuracy

def Vectorization(user_reviews):
    vocab_list = []
    user_dict = {}
    for user in user_reviews:
        for item in user_reviews[user]:
            review_text = user_reviews[user][item][0]
            rating = user_reviews[user][item][1]
            public_info = user_reviews[user][item][2]
            user_vocab_every = []
            for each in review_text:
                word = each[0]
                vocab_list.append(word)
                user_vocab_every.append(word)
            user_dict.setdefault(user,{})
            user_dict[user].setdefault(item, [user_vocab_every,rating,public_info])
    vocab_set = set(vocab_list)
    # print 'user_dict:',user_dict
    # print 'vocab_set_length:',len(vocab_set)
    # print 'vocab_set:',vocab_set
    return vocab_set,user_dict

def TF_IDF(vocab_set,user_dict):
    user_vertorize_dict = {}
    for user in user_dict:
        for item in user_dict[user]:
            user_vertorize_every = []
            user_reviews_list = user_dict[user][item][0]
            rating = user_dict[user][item][1]
            public_info = user_dict[user][item][2]
            for word in vocab_set:
                num = user_reviews_list.count(word)
                #print user,'====',word,'====',num
                user_vertorize_every.append(num)
            user_vertorize_dict.setdefault(user,{})
            user_vertorize_dict[user].setdefault(item,[user_vertorize_every,rating,public_info])
        # print 'user_vertorize_dict:',user_vertorize_dict
        # print 'user_vertorize_dict keys:',user_vertorize_dict.keys()
        # print 'user_vertorize_dict values:',len(user_vertorize_dict.values()[0].keys())
    return user_vertorize_dict

def lda_model(user_vertorize_dict,vocab_set):
    #获取array
    theme_dict = {}
    for user in user_vertorize_dict:
        length = len(user_vertorize_dict[user].keys())
        review_list = []
        rating_list =[]
        public_list = []
        for item in user_vertorize_dict[user]:
            review = user_vertorize_dict[user][item][0]
            rating = user_vertorize_dict[user][item][1]
            public_info = user_vertorize_dict[user][item][1]
            review_list.append(review)
            rating_list.append(rating)
            public_list.append(public_info)
        data = np.array(review_list)
        # print data.shape
        vocab_set = list(vocab_set)
        #新建model
        model = lda.LDA(n_topics=20, n_iter=500, random_state=1)
        model.fit(data)

        # print "####20个主题在vocab上的概率分布..."
        topic_word = model.topic_word_
        # print("type(topic_word): {}".format(type(topic_word)))
        # print("shape: {}".format(topic_word.shape))
        # print(vocab_set[:3])
        # print(topic_word[:, :3])

        # print "####每个主题的概率分布求和..."
        # for n in range(5):
        #     # print topic_word[n, :]
        #     sum_pr = sum(topic_word[n, :])
        #     print("topic: {} sum: {}".format(n, sum_pr))

        print "####每个主题前n个词汇..."
        n = 5
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(vocab_set)[np.argsort(topic_dist)][:-(n + 1):-1]
            print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))

        # print "####每个文档的主题概率分布..."
        doc_topic = model.doc_topic_
        # print("type(doc_topic): {}".format(type(doc_topic)))
        # print("shape: {}".format(doc_topic.shape))

        average_rate = np.array(rating_list).mean()
        m,n = doc_topic.shape
        zeros = np.zeros([1,n])
        for i in range(m):
            rating = 1/(1+e**(average_rate - rating_list[i]))
            zeros += doc_topic[i,:] * rating
        new_theme = zeros/length
        #print "new_theme:",new_theme
        theme_dict[user] = new_theme
    #print theme_dict
    return theme_dict
        # for n in range(10):
        #     topic_most_pr = doc_topic[n].argmax()
        #     print("doc: {} topic: {}".format(n, topic_most_pr))
        #
        # import matplotlib.pyplot as plt
        #
        # f, ax = plt.subplots(5, 1, figsize=(8, 6), sharex=True)
        # for i, k in enumerate([0, 5, 9, 14, 19]):
        #     ax[i].stem(topic_word[k, :], linefmt='b-',
        #                markerfmt='bo', basefmt='w-')
        #     ax[i].set_xlim(-50, 10000)
        #     ax[i].set_ylim(0, 0.15)
        #     ax[i].set_ylabel("Prob")
        #     ax[i].set_title("topic {}".format(k))
        #
        # ax[4].set_xlabel("word")
        #
        # plt.tight_layout()
        # plt.show()
        #
        # import matplotlib.pyplot as plt
        #
        # f, ax = plt.subplots(5, 1, figsize=(8, 6), sharex=True)
        # for i, k in enumerate([1, 3, 4, 8, 9]):
        #     ax[i].stem(doc_topic[k, :], linefmt='r-',
        #                markerfmt='ro', basefmt='w-')
        #     ax[i].set_xlim(-1, 21)
        #     ax[i].set_ylim(0, 1)
        #     ax[i].set_ylabel("Prob")
        #     ax[i].set_title("Document {}".format(k))
        #
        # ax[4].set_xlabel("Topic")
        # plt.tight_layout()
        # plt.show()

# 计算一个用户的平均评分
def getAverageRating(user):
    rating_list =[]
    for item in trainSet[user].keys():
        rating = trainSet[user][item][1]
        rating_list.append(rating)
    average = np.array(rating_list).mean()
    return average

# 计算用户相似度
def getUserSim(theme_dict):
    userSim = {}
    # 计算用户的用户相似度
    for user in theme_dict.keys():  # 对每个用户u
        userSim.setdefault(user, {})  # 将用户u加入userSim中设为key，该用户对应一个字典
        average_user_rate = theme_dict[user].mean()  # 获取用户u对电影的平均评分
        for sam_user in theme_dict.keys():  # 对与用户u相关的每个用户n
            userSim[user].setdefault(sam_user, 0)  # 将用户n加入用户u的字典中
            average_sam_user_rate = theme_dict[sam_user].mean()  # 获取用户n对电影的平均评分
            part1 = (theme_dict[user]-average_user_rate)*(theme_dict[sam_user]-average_sam_user_rate)
            part1 = part1.sum()
            part2 = (theme_dict[user]-average_user_rate)**2
            part2 = sqrt(part2.sum())
            part3 = (theme_dict[sam_user]-average_sam_user_rate)**2
            part3 = sqrt(part3.sum())
            if part2 == 0 or part3 == 0:  # 若分母为0，相似度为0
                userSim[user][sam_user] = 0
            else:
                userSim[user][sam_user] = part1 / (part2 * part3)
    # print "用户相似度列表中的用户数：",len(userSim.keys())
    # print "userSim:",userSim
    # print "一个用户的相似用户数：",len(userSim.items()[0][1].keys())
    return userSim



# 寻找用户最近邻并生成推荐结果
def getRecommendations(N, trainSet, userSim):
    pred = {}
    for user in userSim.keys():  # 对每个用户
        pred.setdefault(user, {})  # 生成预测空列表
        interacted_items = trainSet[user].keys()  # 获取该用户评过分的电影
        average_u_rate = getAverageRating(user)  # 获取该用户的评分平均分
        userSimSum = 0
        simUser = sorted(userSim[user].items(), key=lambda x: x[1], reverse=True)[0:N] #N为推荐人数
        for n, sim in simUser:
            if n == user:continue
            average_n_rate = getAverageRating(n)
            userSimSum += sim  # 对该用户近邻用户相似度求和
            for m, nreview_list in trainSet[n].items():
                if m in interacted_items:
                    continue                #如果相似用户喜欢的电影在目标用户喜欢的列表中，则舍去
                else:
                    nrating = nreview_list[1]
                    pred[user].setdefault(m, 0)
                    pred[user][m] += (sim * (nrating - average_n_rate))  #这里是否可以考虑（X-avg（x）/std（x）
        noeqzero = 0
        for m in pred[user].keys():
            if userSimSum!=0:
                noeqzero+=1
                pred[user][m] = average_u_rate + (pred[user][m] * 1.0) / userSimSum
            else:
                pred[user][m] = average_u_rate
    print "noeqzero:",noeqzero
    #print "pred:",pred.items()[0]
    #print "预测一个用户的评论商品数：",len(pred.items()[0][1].keys())
    #print "pred%s:"%pred.items()[0][0],sorted(pred.items()[0][1].items(),key=lambda x: x[1], reverse=True)
    return pred


#计算recall
def get_recall(testSet, pred):
    test_user = testSet.keys()
    pred_user = pred.keys()
    recommend_user = set(test_user)&set(pred_user[:200])
    dict={}
    for user in recommend_user:  # 对每一个用户
        recommend_prod = set(pred[user].keys()) #取推荐列表前100个商品推荐
        true_prod = set(testSet[user].keys())
        recall = len(recommend_prod&true_prod)/float(len(true_prod))
        dict[user]=recall
    recall_avg = np.array(dict.values()).mean()
    return recall_avg

#计算precision
def get_precision(testSet, pred):
    test_user = testSet.keys()
    pred_user = pred.keys()
    recommend_user = set(test_user)&set(pred_user[:200])
    dict={}
    for user in recommend_user:  # 对每一个用户
        recommend_prod = set(pred[user].keys()) #取推荐列表前100个商品推荐
        true_prod = set(testSet[user].keys())
        if len(recommend_prod) ==0:
            precision=0
        else:
            precision = len(recommend_prod&true_prod)/float(len(recommend_prod))
        dict[user]=precision
    precision_avg = np.array(dict.values()).mean()
    return precision_avg







if __name__ == '__main__':
    print "load data..."
    trainSet, testSet, u2u = loadData()
    read_stopwords()
    print "review text processed..."
    user_reviews, NLTK_character, NLTK_character_accuracy = review_process(trainSet)
    # print "Vectorization..."
    # vocab_set, user_dict = Vectorization(user_reviews)
    # print "TF_IDF..."
    # user_vertorize_dict = TF_IDF(vocab_set, user_dict)
    #
    #print 'user_reviews:',user_reviews.keys()
    print 'first user:',user_reviews.items()[0]
    print '一个用户评论商品数：',len(user_reviews.values()[0].keys())
    #print 'NLTK_character:',NLTK_character
    print 'NLTK_character_accuracy:',np.array(NLTK_character_accuracy).mean()
    # print "create theme dict..."
    # theme_dict = lda_model(user_vertorize_dict, vocab_set)
    #
    # userSim = getUserSim(theme_dict)
    # for N in (10,):  # 对不同的近邻数
    #     print '邻居数为：N=%d：'%(N)
    #     pred = getRecommendations(N, trainSet, userSim)  # 获得推荐
    #     recall = get_recall(testSet, pred)
    #     precision = get_precision(testSet, pred)
    #     F1_score = (2*recall*precision)/(recall+precision)
    #     print 'recall:', recall
    #     print 'precision:', precision
    #     print 'F1_score:',F1_score

