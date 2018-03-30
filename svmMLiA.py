import numpy as np

#读入数据
def loadData(filename):
    dataFile=open(filename)
    dataMat=[]#用于储存训练样本的特征
    label=[]#用于储存样本的分类结果
    for line in dataFile.readlines():
        matStr=line.strip().split('\t')#将输入分割
        dataMat.append([float(matStr[0]),float(matStr[1])])
        label.append(float(matStr[2]))
    return dataMat,label


#用于对α2进行剪切
def clipAlpha(aj,H,L):
    if(aj>H):aj=H
    if(aj<L):aj=L
    return aj

#生成一个随机的内循环变量
def selectJrand(i,m):
    j=i
    while (j==i):
        j=int(np.random.uniform(0,m))
    return j

#SMO简化版算法核心
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix=np.mat(dataMatIn); labelMat=np.mat(classLabels).transpose()
    m,n=np.shape(dataMatrix)
    alpha=np.mat(np.zeros((m,1)))
    b=0;
    iterCount=0
    while (iterCount<maxIter):
        alphaPairsChanged=0
        for i in range(m):
            gI=float(np.multiply(alpha,labelMat).T*(dataMatrix*dataMatrix[i,:].T)+b)
            EI=gI-float(labelMat[i])#计算E1,用于计算α1new及α2new
            if((alpha[i]>0)and(labelMat[i]*EI>toler))or((alpha[i]<C)and(labelMat[i]*EI<-toler)):
            #选择α1的原则：α1不满足KKT条件，且优先选择间隔区域内（离超平面距离小于ξ的支持向量）
                j=selectJrand(i,m)
                gJ=float(np.multiply(alpha,labelMat).T*(dataMatrix*dataMatrix[j,:].T)+b)
                EJ=gJ-float(labelMat[j])#计算E2,用于计算α1new及α2new
                eta=float((dataMatrix[i,:]-dataMatrix[j,:])*(dataMatrix[i,:]-\
                          dataMatrix[j,:]).T)
                #计算η，我利用的是向量点积，而不是展开式
                if (eta<=0):
                    print('eta<=0')
                    continue
                alphaJOld=alpha[j].copy()
                alphaIOld=alpha[i].copy()
                #计算未剪切的α2，
                alpha[j]=alphaJOld+labelMat[j]*(EI-EJ)/eta     
                #确定需要剪切的约束范围，根据y的不同，约束范围也不同
                if(labelMat[i]!=labelMat[j]):
                    L=max(0,alphaJOld-alphaIOld)
                    H=min(C,C+alphaJOld-alphaIOld)
                else:
                    L=max(0,alphaJOld+alphaIOld-C)
                    H=min(C,alphaJOld+alphaIOld)
                #计算剪切后的α2
                alpha[j]=clipAlpha(alpha[j],H,L)
                if(alpha[j]-alphaJOld<0.0001):
                    print('j not moving enough!')
                    continue#如果α2基本没有变化，则跳出当前次循环，找下一个外层循环变量
                      
                #根据α2计算α1
                alpha[i]=alphaIOld+labelMat[i]*labelMat[j]*(alphaJOld-alpha[j])    
                #分别利用α1及α2计算b                
                bInew=float(-EI-labelMat[i]*float(dataMatrix[i,:]*dataMatrix[i,:].T)*\
                            (alpha[i]-alphaIOld)-labelMat[j]*float(dataMatrix[i\
                            ,:]*dataMatrix[j,:].T)*(alpha[j]-alphaJOld)+b)
                bJnew=float(-EJ-labelMat[i]*float(dataMatrix[i,:]*dataMatrix[j,:].T)*\
                            (alpha[i]-alphaIOld)-labelMat[j]*float(dataMatrix[j\
                            ,:]*dataMatrix[j,:].T)*(alpha[j]-alphaJOld)+b)
    
                #根据α1及α2的不同取值确定b
                if (alpha[i]>0)and(alpha[i]<C):  b=bInew
                elif(alpha[j]>0)and(alpha[j]<C): b=bJnew
                else:b=(bInew+bJnew)/2.0
                alphaPairsChanged+=1
                print('iterCount:%d,i:%d,pair changed %d'%(iterCount,i,alphaPairsChanged))
        if(alphaPairsChanged==0):iterCount+=1
        else:iterCount=0
        print('iteration number:%d'%(iterCount))   
    return alpha,b



#用于调试代码
#dataMat,labelMat=loadData('testSet.txt')
#alphas,b=smoSimple(dataMat,labelMat,0.6,0.001,40)