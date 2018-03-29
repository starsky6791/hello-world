import numpy as np

def loadData(filename):
    dataFile=open(filename)
    dataMat=[]
    label=[]
    for line in dataFile.readlines():
        matStr=line.strip().split('\t')
        dataMat.append(float(matStr[0]),float(matStr[1]))
        label.append(float(matStr[2]))
    return dataMat,label

def clipAlpha(aj,H,L):
    if(aj>H):aj=H
    if(aj<L):aj=L
    return aj

def selectJrand(i,m):
    j=i
    while (j==i):
        j=int(np.random.uniform(0,m))
    return j


def E(alphaI,b,yI,dataMatrix):
    gx=np.multiply(alphaI,)
    
    
    return E
#SMO简化版算法核心
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix=np.mat(dataMatIn); labelMat=np.mat(classLabels).transpose()
    m,n=np.shape(dataMatrix)
    alpha=np.mat(np.zeros((m,1)))
    b=0;
    iterCount=0
    while (iterCount<maxIter):
        for i in range(m):
            gI=float(np.multiply(alpha,labelMat).T*(dataMatrix*dataMatrix[i,:])+b)
            EI=gI-float(labelMat[i])#计算E1,用于计算α1new及α2new
            if((alpha[i]>0)and())or((alpha[i]<C)and()):#选择α1的原则：α1不满足KKT条件，且优先选择间隔区域内（离超平面距离小于ξ的支持向量）
                j=selectJrand(i,m)
                gJ=np.multiply(alpha,labelMat).T*(dataMatrix*dataMatrix[j,:])+b
                EJ=gJ-float(labelMat[j])#计算E2,用于计算α1new及α2new
                eta=float((dataMatrix[i,:]-dataMatrix[j,:]).T*(dataMatrix[i,:]-dataMatrix[j,:]))#计算η，我利用的是向量点积，而不是展开式
                alphaJOld=alpha[j]
                alphaIOld=alpha[i]
                alpha[j]=alphaJOld+labelMat[j]*(EI-EJ)/eta                              
                if(labelMat[i]!=labelMat[j]):
                    L=max(0,alphaJOld-alphaIOld)
                    H=min(C,C+alphaJOld-alphaIOld)
                else:
                    L=max(0,alphaJOld+alphaIOld-C)
                    H=min(C,alphaJOld+alphaIOld)
                alpha[j]=clipAlpha(alpha[j],H,L)
                alpha[i]=alphaIOld+labelMat[i]*labelMat[j]*(alphaJOld-alpha[j])                        
                bInew=-EI-labelMat[i]*float(dataMat[i,:].T*dataMat[i,:])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    
    i
    
    
    
    
    return alpha,b