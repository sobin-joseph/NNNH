#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:




# In[9]:



#print(likelihoodMu(tseries1))


mus1=[]
mus2=[]
musC=[]
mus1grad=[]
mus2grad=[]
musCgrad=[]

dictMus={}
dictIntegrate={}

dictgradient={}
epsilon = 1e-8
def nnIntializeMus(nNeurons1,tseries):
    global tmax
    mus1.clear()
    mus2.clear()
    musC.clear()
    mus1grad.clear()
    mus2grad.clear()
    musCgrad.clear()
    
    c=np.random.uniform(0,1,1)[0]*0.01
    alphas=(np.random.uniform(0,1,nNeurons1)).reshape(-1,1)*0.2
    alpha0=np.random.uniform(0,1,1)*0
    
    betas1=(np.random.uniform(0,1,int(nNeurons1/2)))*0.001
    betas2=(np.random.uniform(-1,0,int(nNeurons1/2)))*0.001

    betas=np.concatenate((betas1,betas2))
    betas=(betas).reshape(-1,1)
    tmax=tseries[-1]
    beta01=(np.random.uniform(-1,0,int(nNeurons1/2)))
    beta02=(np.random.uniform(0,1,int(nNeurons1/2)))
    beta0=(np.concatenate((beta01,beta02)))
    beta0=(beta0).reshape(-1,1)
    
    
    
                                        
    
    #mean=np.mean(tseries)
    #std=np.std(tseries)
    
    #betas=weight_relu.reshape(-1,1)
    #beta0=bias_relu.reshape(-1,1)
    #alphas=weight_exp.reshape(-1,1)
    #alpha0=bias_exp-c
    
    
    betas=betas*len(tseries)/tmax
  
    #beta0=beta0-betas*mean
    
    mus1.append(alphas)
    mus1.append(alpha0)
    mus2.append(betas)
    mus2.append(beta0)
    musC.append(c)
    
    a1=np.zeros((len(alphas),1))
    a2=0
    mus1grad.append(a1)
    mus1grad.append(a2)
    mus2grad.append(a1)
    mus2grad.append(a1)
    musCgrad.append(0)

    
    return
def nnMufunction(x):
    alphas=mus1[0]
    alpha0=mus1[1]
    betas=mus2[0]
    beta0=mus2[1]
    c=musC[0]
    x=np.array(x)
    x=x.reshape(-1)
    n1=np.maximum(betas*x.reshape(1,-1)+beta0,0)
    n2=np.dot(alphas.T,n1)+alpha0
    y=(np.maximum(n2,0))+c
    return y.reshape(-1)



def outerInflections():
    alphas=mus1[0]
    alpha0=mus1[1]
    betas=mus2[0]
    beta0=mus2[1]
    inflection=[]
    inflectionPs=dictMus['innerinflection']
    infl1=-1000
    for i in range(1,len(inflectionPs)):
        iP1=inflectionPs[i]
        iP2=inflectionPs[i-1]
        n1=betas*(iP1-epsilon)+beta0
        dn1=(n1>0)
        n2=(alphas*dn1*n1).sum()+alpha0
        dn2=(n2>0)
        v1=(alphas*dn1*beta0).sum()
        v2=(alphas*dn1*betas).sum()
        infl=(-alpha0[0]-v1)/v2
        if (infl1!=infl) &(iP1>infl)&(iP2<infl):
            inflection.append(infl)
            infl1=infl
    inflection=np.array(inflection)
    inflection=np.sort(inflection)
    inflection=inflection[inflection<tmax]
    dictMus['outerinflection']=inflection
    
    return 



def inflectionPoints():
    
    dictMus.clear()
 
    den=mus2[0]+epsilon*(mus2[0]==0)
    x=-mus2[1]/mus2[0]
    x=x[x>0]
    x=np.sort(x)
    interestX=np.append(0,x)
    inflectionPs=interestX
    
    inflectionPs=np.sort(inflectionPs)
    inflectionPs=inflectionPs[inflectionPs<tmax]
    dictMus['innerinflection']= inflectionPs
    outerInflections()
    outerInflections1=dictMus['outerinflection']
    inflections=np.concatenate((inflectionPs,outerInflections1))
    inflections=np.sort(inflections)
    dictMus['inflection']=inflections
    
    return



def nnPrecalculatemus():
    dictIntegrate.clear()
    alphas=mus1[0]
    alpha0=mus1[1]
    betas=mus2[0]
    beta0=mus2[1]
    c=musC[0]
    inflectionPs=dictMus['inflection']
    dictIntegrate[0]=0
    Integr1=0
    for j in range(1,len(inflectionPs)):
        iP1=inflectionPs[j]
        iP2=inflectionPs[j-1]
        n1=betas*(iP1-epsilon)+beta0
        dn1=(n1>0)
        n2=(alphas*n1*dn1).sum()+alpha0
        dn2=(n2>0)
        term3=(alphas*iP1*((betas*iP1*0.5)+beta0)*dn1).sum()+alpha0*iP1
        term4=(alphas*iP2*((betas*iP2*0.5)+beta0)*dn1).sum()+alpha0*iP2
        Integr1=Integr1+(term3-term4)*dn2+c*(iP1-iP2)
        dictIntegrate[iP1]=Integr1.sum()
    return
       

def nnIntegratedmus(iArray,tseries):
    y=np.zeros(len(iArray))
    nnPrecalculatemus()
    count=0
    for j in np.nditer(iArray):
        tj=tseries[j]
        if tj>0:
            iP=tseries[j-1]
            integral1=nnIntegratedMusPart(tj)
            integral2=nnIntegratedMusPart(iP)
            y[count]=integral1-integral2
        count+=1
    return y
        

    

 
    
def nnIntegratedMusPart(x):
    x=x
    alphas=mus1[0]
    alpha0=mus1[1]
    betas=mus2[0]
    beta0=mus2[1]
    c=musC[0]
    inflectionPs=dictMus['inflection']

    iP=(max(inflectionPs[inflectionPs<=x]))
    previousVal=dictIntegrate.get(iP)
    n1=betas*(x-epsilon)+beta0
    dn1=(n1>0)
    n2=(alphas*n1*dn1).sum()+alpha0
    dn2=(n2>0)

    term3=(alphas*x*((betas*x*0.5)+beta0)*dn1).sum()+alpha0*x
    term4=(alphas*iP*((betas*iP*0.5)+beta0)*dn1).sum()+alpha0*iP
    Integr1=(term3-term4)*dn2+c*(x-iP)+previousVal
    return Integr1
    
def nnLikelihoodmu(iArray,tseries):

    ll=(nnIntegratedmus(iArray,tseries)).sum()
    print("integratedKErnel",ll)
    for j in np.nditer(iArray):
        tj=tseries[j]
        if (tj>0):
            kernel=nnMufunction(tj)
            ll=ll-np.log((kernel.sum()))
    return ll


def plotMus(tseries):
    nn_y=nnMufunction(tseries)
    plt.plot(tseries,nn_y)
    plt.pause(0.0005)
    
    return

def nnSGDMus(lr_mu,nEpochs,tseries,nNeurons=50):
    
    tmax=tseries[-1]
    nnIntializeMus(nNeurons,tseries)


    inflectionPoints()
    lr2_mu=lr_mu*1e-3
    shouldPrint=0
    beta_1 = 0.9
    beta_2 =0.999
    epsilon = 1e-8
    count = 0
    bestll = 1e8
    m_t_A = np.zeros((len(mus1[0]),1))*0
    v_t_A = np.zeros((len(mus1[0]),1))*0
    m_t_A0 = 0
    v_t_A0 = 0
    m_t_B = np.zeros((len(mus1[0]),1))*0
    v_t_B = np.zeros((len(mus1[0]),1))*0
    m_t_B0 = np.zeros((len(mus1[0]),1))*0
    v_t_B0 = np.zeros((len(mus1[0]),1))*0
    m_t_C=0
    v_t_C=0
    optimalParams=[mus1,mus2,musC]
    initmus1=np.array(mus1)
    initmus2 = np.array(mus2)
    likelihoodCurve=[]
    #likelihood=nnLikelihoodmu(np.arange(0,len(tseries),1))
    for j in range(0,nEpochs+1,1):
        rsample= np.random.choice(len(tseries),10000,replace = True)
        for i in range(0,len(rsample),500):
            gradientMus1(rsample[i:i+500],tseries)
            #discr_case=[mus1grad[0],mus1grad[1],mus2grad[0],mus2grad[1]]
            #gradientMus1(rsample[i:i+500])
            #symb_case=[mus1grad[0],mus1grad[1],mus2grad[0],mus2grad[1]]
            #likelihood=nnLikelihoodmu(np.arange(0,len(tseries),1))
            #print('Likelihood before is', likelihood)
            count=count+1 
            if(shouldPrint):
                print('Alpha values', mus1[0])
                print('gradient alpha values', mus1grad[0])
            m_t_A = beta_1*m_t_A + (1-beta_1)*mus1grad[0]	#updates the moving averages of the gradient
            v_t_A = beta_2*v_t_A + (1-beta_2)*(mus1grad[0]*mus1grad[0])	#updates the moving averages of the squared gradient
            m_cap_A = m_t_A/(1-(beta_1**count))		#calculates the bias-corrected estimates
            v_cap_A = v_t_A/(1-(beta_2**count))		#calculates the bias-corrected estimates
            mus1[0]= mus1[0]-(lr_mu*m_cap_A)/(np.sqrt(v_cap_A)+epsilon)
            #print('Alpha values', mus1[0]-initmus1[0])
            if(shouldPrint):
                print('Alpha0 values', mus1[1])
                print('gradient alpha0 values', mus1grad[1])
            m_t_A0 = beta_1*m_t_A0 + (1-beta_1)*mus1grad[1]	#updates the moving averages of the gradient
            v_t_A0 = beta_2*v_t_A0 + (1-beta_2)*(mus1grad[1]*mus1grad[1])	#updates the moving averages of the squared gradient
            m_cap_A0 = m_t_A0/(1-(beta_1**count))		#calculates the bias-corrected estimates
            v_cap_A0 = v_t_A0/(1-(beta_2**count))		#calculates the bias-corrected estimates
            mus1[1]= mus1[1]-(lr_mu*m_cap_A0)/(np.sqrt(v_cap_A0)+epsilon)
            #print('Alpha- values', mus1[1]-initmus1[1])
            if(shouldPrint):
                print('Beta values', mus2[0])
                print('gradient values', mus2grad[0])
            m_t_B = beta_1*m_t_B + (1-beta_1)*mus2grad[0]	#updates the moving averages of the gradient
            v_t_B = beta_2*v_t_B + (1-beta_2)*(mus2grad[0]*mus2grad[0])	#updates the moving averages of the squared gradient
            m_cap_B = m_t_B/(1-(beta_1**count))		#calculates the bias-corrected estimates
            v_cap_B = v_t_B/(1-(beta_2**count))		#calculates the bias-corrected estimates
            mus2[0]= mus2[0]-(lr2_mu*m_cap_B)/(np.sqrt(v_cap_B)+epsilon)
            #print('Beta- values', mus2[0]-initmus2[0])
            if(shouldPrint): 
                print('Beta0 values', mus2[1])
                print('gradient0 values', mus2grad[1])
            m_t_B0 = beta_1*m_t_B0 + (1-beta_1)*mus2grad[1]	#updates the moving averages of the gradient
            v_t_B0 = beta_2*v_t_B0 + (1-beta_2)*(mus2grad[1]*mus2grad[1])	#updates the moving averages of the squared gradient
            m_cap_B0 = m_t_B0/(1-(beta_1**count))		#calculates the bias-corrected estimates
            v_cap_B0 = v_t_B0/(1-(beta_2**count))		#calculates the bias-corrected estimates
            mus2[1]= mus2[1]-(lr_mu*0.1*m_cap_B0)/(np.sqrt(v_cap_B0)+epsilon)
            #print('Beta0- values', mus2[1]-initmus2[1])
            
            m_t_C = beta_1*m_t_C + (1-beta_1)*musCgrad[0]	#updates the moving averages of the gradient
            v_t_C = beta_2*v_t_C + (1-beta_2)*(musCgrad[0]*musCgrad[0])	#updates the moving averages of the squared gradient
            m_cap_C = m_t_C/(1-(beta_1**count))		#calculates the bias-corrected estimates
            v_cap_C = v_t_C/(1-(beta_2**count))		#calculates the bias-corrected estimates
            musC[0]= musC[0]-(lr_mu*m_cap_C)/(np.sqrt(v_cap_C)+epsilon)
            musC[0]=max(1e-4,musC[0])
            #print("c value", musC[0])
            
            
            #likelihood=likelihoodsample(np.arange(0,len(tseries),1))
            #print('Likelihood after is', likelihood)
            inflectionPoints()
            #print("alpha",symb_case[0]-discr_case[0])
            #print("betas",symb_case[2]-discr_case[2])
            #print("beta0",symb_case[3]-discr_case[3])
        #likelihood=nnLikelihoodmu(np.arange(0,len(tseries),1),tseries)
        #likelihood1=likelihoodsample(np.arange(0,len(tseries),1))
        #if likelihood<bestll:
        #    bestll=likelihood
        #    optimalParams=[mus1,mus2,musC]
        #print(i,j,likelihood)
    likelihood=nnLikelihoodmu(np.arange(0,len(tseries),1),tseries)
    print(likelihood)
        
        #likelihoodCurve.append(likelihood)
    #plotMus(tseries)
    #plt.plot(likelihoodCurve)
    return optimalParams



#likelihood1=likelihoodsample(np.arange(0,len(tseries),1))
#print(likelihood1)
    




# In[5]:



def gradientMus1(iArray,tseries):
    alphas=mus1[0]
    alpha0=mus1[1]
    betas=mus2[0]
    beta0=mus2[1]
    c=musC[0]
    gradA=np.zeros((len(alphas),1))*0
    gradB=np.zeros((len(alphas),1))*0
    gradB0=np.zeros((len(alphas),1))*0
    gradA0=0
    gradC=0
    preIntGradients()


    
    for j in np.nditer(iArray):
        tj=tseries[j]
        if tj>0.0000000:
            iP=tseries[j-1]
            IntegratMu1=gradientMusPart(tj)
            IntegratMu2=gradientMusPart(iP)
            gradA=gradA+(IntegratMu1[0]-IntegratMu2[0])
            gradA0=gradA0+(IntegratMu1[1]-IntegratMu2[1])
            gradB=gradB+(IntegratMu1[2]-IntegratMu2[2])
            gradB0=gradB0+(IntegratMu1[3]-IntegratMu2[3])
            gradC=gradC+(IntegratMu1[4]-IntegratMu2[4])
            logpart=((nnMufunction(tj)).sum())
            
            inverselog=1/logpart
            n1=np.maximum(betas*(tj)+beta0,0)
            dn1=(n1>0)
            term1=(alphas*n1).sum()+alpha0
            dn2=(term1>0)
            
            gradA=gradA-(n1)*dn2*inverselog
            gradA0=gradA0-(1)*dn2*inverselog
            gradB=gradB-(alphas*tj*dn1)*dn2*inverselog
            gradB0=gradB0-(alphas*dn1)*dn2*inverselog
            gradC=gradC-(1)*inverselog
    length=len(iArray)
    
    gradA,gradA0,gradB,gradB0=gradA/length,gradA0/length,gradB/length,gradB0/length
    mus1grad[0]=gradA
    mus1grad[1]=gradA0
    mus2grad[0]=gradB
    mus2grad[1]=gradB0
    musCgrad[0]=gradC
    return
            
    
def gradientMusPart(x):
    alphas=mus1[0]
    alpha0=mus1[1]
    betas=mus2[0]
    beta0=mus2[1]
    c=musC[0]
    gradA=np.zeros((len(alphas),1))*0
    gradB=np.zeros((len(alphas),1))*0
    gradB0=np.zeros((len(alphas),1))*0
    gradA0=0
    gradC=0
    inflectionPs=dictMus['inflection']
 
    
    tj=x
    iP=(max(inflectionPs[inflectionPs<=tj]))
    gradAP,gradA0P,gradBP,gradB0P,gradCP=dictgradient.get(iP)
    
    n1=betas*(tj-epsilon)+beta0
    dn1=(n1>0)
    n2=(alphas*dn1*n1).sum()+alpha0
    dn2=(n2>0)
    
    

    cn1=tj*(0.5*betas*tj+beta0)*dn1
    cn2=iP*(0.5*betas*iP+beta0)*dn1
    gradA0=gradA0+gradA0P+(tj-iP)*dn2
    gradA=gradA+gradAP+(cn1-cn2)*dn2
    gradB=gradB+gradBP+(0.5*alphas*dn1*(tj**2-iP**2))*dn2
    gradB0=gradB0+gradB0P+alphas*dn1*(tj-iP)*dn2
    gradC=gradC+gradCP+(tj-iP)

    return gradA,gradA0,gradB,gradB0,gradC
        
    
def preIntGradients():
    dictgradient.clear()
    alphas=mus1[0]
    alpha0=mus1[1]
    betas=mus2[0]
    beta0=mus2[1]
    c=musC[0]
    gradA=np.zeros((len(alphas),1))*0
    gradB=np.zeros((len(alphas),1))*0
    gradB0=np.zeros((len(alphas),1))*0
    gradA0=0
    gradC=0
   
    inflectionPs=dictMus['inflection']
  
    dictgradient[0.0]=[gradA*0,gradA0*0,gradB*0,gradB0*0,gradC*0]
    
    for j in range(1,len(inflectionPs)):
        iP1=inflectionPs[j]
        iP2=inflectionPs[j-1]
        n1=betas*(iP1-epsilon)+beta0
        dn1=(n1>0)
        n2=(alphas*dn1*n1).sum()+alpha0
        dn2=(n2>0)

        cn1=iP1*(0.5*betas*iP1+beta0)*dn1
        cn2=iP2*(0.5*betas*iP2+beta0)*dn1
        gradA0=gradA0+(iP1-iP2)*dn2
        gradA=gradA+(cn1-cn2)*dn2
        gradB=gradB+0.5*alphas*dn1*(iP1**2-iP2**2)*dn2
        gradB0=gradB0+alphas*dn1*(iP1-iP2)*dn2
        
        gradC=gradC+(iP1-iP2)
        dictgradient[iP1]=[gradA,gradA0,gradB,gradB0,gradC]
    return
        
    
    


# In[ ]:




