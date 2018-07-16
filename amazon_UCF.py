#encoding=utf-8
from math import sqrt
import pandas as pd
import numpy as np

def loadData():
    trainSet = {}
    testSet = {}
    trainprodUser = {}
    testprodUser = {}
    u2u = {}

    TrainFile = 'E:\major\paper\data\\train.csv'  # 指定训练集
    TestFile = 'E:\major\paper\data\\test.csv'  # 指定测试集
    # 加载训练集
    traindata = pd.read_csv(TrainFile)
    #print traindata.columns
    userlist = list(traindata['REVIEWER_ID'])
    itemlist = list(traindata['ASIN'])
    ratelist = list(traindata['OVERALL'])
    timelist = list(traindata['UNIX_REVIEW_TIME'])
    m = len(userlist)
    for i in range(m):
        userId = userlist[i]
        itemId = itemlist[i]
        rating = ratelist[i]
        timestamp = timelist[i]
        trainSet.setdefault(userId, {})
        trainSet[userId].setdefault(itemId, float(rating))
        #trainSet记录每个用户对每个电影的评分，形如：{'userid':{'itemid':rating}}
        trainprodUser.setdefault(itemId, [])
        trainprodUser[itemId].append(userId.strip())
        #trainprodUser记录每个电影的评论用户id,形如：{'itemid':['userid1','userid2']}
    print "trainSet:",trainSet.items()[0]
    print "trainprodUser",trainprodUser.items()[0]
    print "训练集用户数：",len(trainSet.keys())
    print "训练集电影数：",len(trainprodUser.keys())
    # 加载测试集
    testdata = pd.read_csv(TestFile)
    #print testdata.columns
    userlist = list(testdata['REVIEWER_ID'])
    itemlist = list(testdata['ASIN'])
    ratelist = list(testdata['OVERALL'])
    timelist = list(testdata['UNIX_REVIEW_TIME'])
    m = len(userlist)
    for i in range(m):
        userId = userlist[i]
        itemId = itemlist[i]
        rating = ratelist[i]
        timestamp = timelist[i]
        testSet.setdefault(userId, {})
        testSet[userId].setdefault(itemId, float(rating))

        testprodUser.setdefault(itemId, [])
        testprodUser[itemId].append(userId.strip())
    print "测试集用户数：",len(testSet.keys())
    print '测试集商品数', len(testprodUser.keys())
    print "testSet:",testSet.items()[0]

    # #判断测试集里的用户数是否都在训练集中
    # num=0
    # for i in trainSet.keys():
    #     for j in testSet.keys():
    #         if i==j:
    #             num+=1
    # print "测试集中的用户在训练集中存在的数量:",num

    # 生成用户共有电影矩阵
    for m in trainprodUser.keys(): #m=itemid
        for u in trainprodUser[m]: #trainprodUser[m]为用户列表，u=userid
            u2u.setdefault(u, {})
            for n in trainprodUser[m]:
                if u != n:
                    u2u[u].setdefault(n, [])
                    u2u[u][n].append(m)
    print "训练集u2u:",u2u.items()[0]
    print "训练集有公共电影的用户数：",len(u2u.keys())
    return trainSet, testSet, u2u

# 计算一个用户的平均评分
def getAverageRating(user):
    average = (sum(trainSet[user].values()) * 1.0) / len(trainSet[user].keys())
    return average

# 计算用户相似度
def getUserSim(u2u, trainSet):
    userSim = {}
    # 计算用户的用户相似度
    for u in u2u.keys():  # 对每个用户u
        userSim.setdefault(u, {})  # 将用户u加入userSim中设为key，该用户对应一个字典
        average_u_rate = getAverageRating(u)  # 获取用户u对电影的平均评分
        for n in u2u[u].keys():  # 对与用户u相关的每个用户n
            userSim[u].setdefault(n, 0)  # 将用户n加入用户u的字典中

            average_n_rate = getAverageRating(n)  # 获取用户n对电影的平均评分

            part1 = 0  # 皮尔逊相关系数的分子部分
            part2 = 0  # 皮尔逊相关系数的分母的一部分
            part3 = 0  # 皮尔逊相关系数的分母的一部分
            for m in u2u[u][n]:  # 对用户u和用户n的共有的每个电影
                part1 += (trainSet[u][m] - average_u_rate) * (trainSet[n][m] - average_n_rate) * 1.0
                part2 += pow(trainSet[u][m] - average_u_rate, 2) * 1.0
                part3 += pow(trainSet[n][m] - average_n_rate, 2) * 1.0

            part2 = sqrt(part2)
            part3 = sqrt(part3)
            if part2 == 0 or part3 == 0:  # 若分母为0，相似度为0
                userSim[u][n] = 0
            else:
                userSim[u][n] = part1 / (part2 * part3)
    print "用户相似度列表中的用户数：",len(userSim.keys())
    print "userSim:",userSim.items()[0]
    print "一个用户的相似用户数：",len(userSim.items()[0][1].keys())
    return userSim


# 寻找用户最近邻并生成推荐结果
def getRecommendations(N, trainSet, userSim):
    pred = {}
    for user in trainSet.keys():  # 对每个用户
        pred.setdefault(user, {})  # 生成预测空列表
        interacted_items = trainSet[user].keys()  # 获取该用户评过分的电影
        average_u_rate = getAverageRating(user)  # 获取该用户的评分平均分
        userSimSum = 0
        simUser = sorted(userSim[user].items(), key=lambda x: x[1], reverse=True)[0:N] #N为推荐人数
        for n, sim in simUser:
            average_n_rate = getAverageRating(n)
            userSimSum += sim  # 对该用户近邻用户相似度求和
            for m, nrating in trainSet[n].items():
                if m in interacted_items:
                    continue                #如果相似用户喜欢的电影在目标用户喜欢的列表中，则舍去
                else:
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


# 计算预测分析准确度
def getMAE(testSet, pred):
    MAE = 0
    rSum = 0
    setSum = 0
    for user in pred.keys():  # 对每一个用户
        for prod, rating in pred[user].items():  # 对该用户预测的每一个电影
            if user in testSet.keys() and prod in testSet[user].keys():  # 如果用户为该电影评过分
                setSum = setSum + 1  # 预测准确数量+1
                rSum = rSum + abs(testSet[user][prod] - rating)  # 累计预测评分误差
    MAE = rSum / setSum
    return MAE

#计算recall
def get_recall(testSet, pred):
    test_user = testSet.keys()
    pred_user = pred.keys()
    recommend_user = set(test_user)&set(pred_user)
    dict={}
    for user in recommend_user:  # 对每一个用户
        recommend_prod = set(pred[user].keys()[:200]) #取推荐列表前100个商品推荐
        true_prod = set(testSet[user].keys())
        recall = len(recommend_prod&true_prod)/float(len(true_prod))
        dict[user]=recall
    recall_avg = np.array(dict.values()).mean()
    return recall_avg

#计算precision
def get_precision(testSet, pred):
    test_user = testSet.keys()
    pred_user = pred.keys()
    recommend_user = set(test_user)&set(pred_user)
    dict={}
    for user in recommend_user:  # 对每一个用户
        recommend_prod = set(pred[user].keys()[:200]) #取推荐列表前100个商品推荐
        true_prod = set(testSet[user].keys())
        if len(recommend_prod) ==0:
            precision=0
        else:
            precision = len(recommend_prod&true_prod)/float(len(recommend_prod))
        dict[user]=precision
    precision_avg = np.array(dict.values()).mean()
    return precision_avg


if __name__ == '__main__':

    print u'正在加载数据...'
    trainSet, testSet, u2u = loadData()

    print u'正在计算用户间相似度...'
    userSim = getUserSim(u2u, trainSet)

    print u'正在寻找最近邻...'
    maelist = []
    neighbor = []
    for N in (10, 30, 50, 70, 90, 110):  # 对不同的近邻数
        print '邻居数为：N=%d：'%(N)
        pred = getRecommendations(N, trainSet, userSim)  # 获得推荐
        #mae = getMAE(testSet, pred)  # 计算MAE

        neighbor.append(N)
        #maelist.append(mae)
        recall = get_recall(testSet, pred)
        precision = get_precision(testSet, pred)
        F1_score = (2*recall*precision)/(recall+precision)
        print 'recall:', recall
        print 'precision:', precision
        print 'F1_score:',F1_score
        #print u'邻居数为：N= %d 时 预测评分准确度为：MAE=%f' % (N, mae)

    # #画图
    # from matplotlib import pyplot as plt
    # plt.plot(neighbor,maelist,'x--','r')
    # plt.show()