def loadData(filename):
    dataFile=open(filename)
    x1=[]
    x2=[]
    label=[]
    for line in dataFile.readlines():
        matStr=line.strip().split('\t')
        x1.append(float(matStr[0]))
        x2.append(float(matStr[1]))
        label.append(float(matStr[2]))
    return x1,x2,label

def clipAlpha(aj,H,L):
    if(aj>H):aj=H
    if(aj<L):aj=L
    return aj

def selectJrand(i,m):
    