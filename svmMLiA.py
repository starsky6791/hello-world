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




#==============================================================================
   
    #以下为完整版SMO算法

#==============================================================================
#用于储存所有数据，包括惩罚因子、误差、输入样本、输入样本的标签及需要最优化的参数α、b
class optStruct:
    def __init__(self,dataMat,labelMat,C,toler):
        self.dataMat=dataMat
        self.labelMat=labelMat
        self.C=C
        self.toler=toler
        self.m=dataMat.shape[0]        
        self.alphas=np.mat(np.zeros([self.m,1]))
        self.b=0
        self.ELabel=np.mat(np.zeros([self.m,2]))#第一列用于表示是否有效的标志，第二列给出E的值

#用于计算E
def calEK(oS,k):
    E=float(np.multiply(oS.alphas,oS.labelMat).T*(oS.dataMat*oS.dataMat[k,:].T)+oS.b)\
            -float(oS.labelMat[k])
    return E

#用于选择α2
def selectJ(i,oS,EI):
    maxdeltaE=0
    maxJ=-1
    EJ=0
    for j in range(oS.m):
        if j!=i:
            EJ= calEK(oS,j)
            if abs(EJ-EI)>maxdeltaE:
                maxdeltaE=abs(EJ-EI)
                maxJ=j
    return maxJ,EJ

#用于更新E值
def updataE(oS,k):
    E=calEK(oS,k)
    oS.eCache[k]=[1,E]


#选择内部循环的节点，并进行α值的更新，返回值为1时说明进行了更新，否则返回0
def innerL(i,oS):
    #计算外层
    EI=calEK(oS,i)
    #不满足KKT条件时选择α
    if((oS.alphas[i]>0)and(oS.labelMat[i]*EI>oS.toler))or\
        ((oS.alphas[i]<oS.C)and(oS.labelMat[i]*EI<-oS.toler)):
                      
            j,EJ=selectJ(i,oS,EI)
            alphaIOld=oS.alphas[i].copy()
            alphaJOld=oS.alphas[j].copy()
            #根据约束条件确定上下界
            if (oS.labelMat[i] != oS.labelMat[j]):
                L = max(0, oS.alphas[j] - oS.alphas[i])
                H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
            else:
                L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
                H = min(oS.C, oS.alphas[j] + oS.alphas[i])
            if L==H: 
                print ("L==H"); 
                return 0
            eta=float((oS.dataMat[i,:]-oS.dataMat[j,:])*(oS.dataMat[i,:]-\
                          oS.dataMat[j,:]).T)
            if eta<=0:print("eta<=0");return 0
            oS.alphas[j]+=oS.labelMat[j]*(EI-EJ)/eta
            oS.alphas[j]=clipAlpha(oS.alphas[j],H,L)
            #updataE(oS,j)
            #计算新的b值
            bInew=float(-EI-oS.labelMat[i]*float(oS.dataMat[i,:]*oS.dataMat[i,:].T)*\
                            (oS.alphas[i]-alphaIOld)-oS.labelMat[j]*float(oS.dataMat[i\
                            ,:]*oS.dataMat[j,:].T)*(oS.alphas[j]-alphaJOld)+oS.b)
            bJnew=float(-EJ-oS.labelMat[i]*float(oS.dataMat[i,:]*oS.dataMat[j,:].T)*\
                            (oS.alphas[i]-alphaIOld)-oS.labelMat[j]*float(oS.dataMat[j\
                            ,:]*oS.dataMat[j,:].T)*(oS.alphas[j]-alphaJOld)+oS.b)
            #根据α的结果选择b值
            if (oS.alphas[i]>0)and(oS.alphas[i]<oS.C):  oS.b=bInew
            elif(oS.alphas[j]>0)and(oS.alphas[j]<oS.C): oS.b=bJnew
            else:oS.b=(bInew+bJnew)/2.0   
            return 1
    else: return 0

#SMO算法外部循环，用于选择外部循环的序号
def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
    #将数据生成oS对象
    oS=optStruct(np.mat(dataMatIn),np.mat(classLabels).T,C,toler)
    iterCount=0
    #遍历整个α的标签，True时遍历整个α
    entireSet=True
    alphaPairsChanged=0
    #迭代条件：为迭代次数小于最大迭代次数且每次迭代后进行过α值修正
    #由于初始时将所有α均设置为0，因此一开始是遍历整个α集合，而不是从边界上进行遍历
    while(iterCount<=maxIter)and((alphaPairsChanged==0)or(entireSet)):
        #迭代前将α的修正次数置为0
        alphaPairsChanged=0
        if (entireSet):#遍历整个集合
            for i in range(oS.m):
                alphaPairsChanged+=innerL(i,oS )
                
                print('遍历了整个集合，迭代次数为：%d ,修改了第 %d 个α值，修改了 %d 次'%(iterCount,i,alphaPairsChanged))
            iterCount+=1 #遍历一次集合迭代次数加1
        else:
            nonBoundSet=np.nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            for i in nonBoundSet:
                alphaPairsChanged+=innerL(i,oS)
               
                print('遍历了内部点，迭代次数为：%d i,修改了第 %d 个α值，修改了 %d 次'%(iterCount,i,alphaPairsChanged))
            iterCount+=1 #遍历一次内部点迭代次数加1
        #由于α初始值为0，所以初始时必定是遍历整个集合
        if entireSet:entireSet=False  #遍历完整个集合后，开始遍历内部点。
        #当entireSet为假时遍历内部点，如果所有内部点都没有进行值更新，则重新遍历整个集合
        elif(alphaPairsChanged==0):entireSet=True
        print('迭代次数为：%d'% iterCount)
    return oS.b,oS.alphas


def calcW(alphas,data,labelDat):
    alphasMat=np.mat(alphas)
    dataMat=np.mat(data)
    labelMat=np.mat(labelDat)
    w=np.multiply(alphasMat,labelMat.T).T*dataMat   
    return w

#用于调试代码
#dataMat,labelMat=loadData('testSet.txt')
#alphas,b=smoP(dataMat,labelMat,0.6,0.001,40)