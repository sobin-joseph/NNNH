#!/usr/bin/env python
# coding: utf-8

# In[2]:


from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import numpy as np
from tick.base import TimeFunction
get_ipython().run_line_magic('matplotlib', 'inline')
from tick.plot import plot_point_process
from tick.hawkes import SimuInhomogeneousPoisson, HawkesKernelTimeFunc, SimuHawkes


# # simulation and Loglikelihood NHP

# In[4]:


ker=2
#a=0.5
#b=0.001

a=0.0004
b=1000
c=1.1
d=0.4

#parabola
#a=0.0002*1e-3
#b=2000
#c=1.5

mu_t=0.4
def mufunction(x,muType):
    if muType==0:
        mu=a*np.exp(-b*x)
    elif muType==1:
        mu=a*(c*x-b)**2
    elif muType==2:
        x=(np.array([x])).reshape(-1)
        mu=mu_t*np.ones(len(x))
    else:
        mu=d*(np.sin(a*2*np.pi*x-b)+c)
    
    
    return mu

def nonHomogenousSimulation(T,muType):
    time_intervals=[]
    s=0
    lambdas=mufunction(np.arange(0,T,0.1))
    max_lambda=max(lambdas)
    time_intervals.append(s)
    while(s<T):
        prob=np.random.uniform(0,1,1)[0]
        interArriT=-np.log(prob)/max_lambda
        s=s+interArriT
        lambdas=mufunction(s)
        prob_thin=np.random.uniform(0,1,1)[0]
        if(prob_thin*max_lambda<=lambdas):
            time_intervals.append(s)
    return np.array(time_intervals)

def likelihoodMu(x,muType):
    if muType==0:
        integratedMu=a*(1-np.exp(-b*x[-1]))/b
    elif muType==1:
        integratedMu=a/(3*c)*((c*x[-1]-b)**3-(c*x[0]-b)**3)
        
    else:
        y1=d*(-np.cos(a*2*np.pi*x[-1]-b)/(np.pi*2*a)+c*x[-1])
        y2=d*(-np.cos((a*2*np.pi*x[0]-b))/(2*a*np.pi)+c*x[0])
        integratedMu=y1-y2
    #print(integratedMu)    
    logKernel=(np.log(mufunction(x))).sum()
    likelihood=(integratedMu).sum()-logKernel
    return likelihood


# ## Simulation and Likelihood function for exponential kernel, Rectangular Kernel, Multplicative Kernel and negative Kernel

# In[5]:



#Exponential Kernel Simulation
def simu_univariate(alpha,beta,mu,T):
    np.random.seed(seed=None)
    lambda_int=mu
    s=0
    tau=[]
    tau.append(0)
    lambda_val=[]
    time_value=[]
    time_value.append(0)
    lambda_val.append(lambda_int+alpha*np.sum(np.exp(-beta*(s-tau[i])) for i in range(len(tau))))
    while s<T:
        sum=0
        new_tau = tau[::-1]
        for i in range(len(new_tau)):
            sum = sum + np.exp(-beta * (s - new_tau[i]))
            if np.exp(-beta*(s-new_tau[i]))<=0.001:
                break
        lambda_ = lambda_int + alpha * sum

        u=np.random.uniform(0,1,1)[0]
        w=-np.log(u)/lambda_
        s=s+w
        d=np.random.uniform(0,1,1)[0]
        sum = 0

        for i in range(len(new_tau)):
            sum = sum + np.exp(-beta * (s - new_tau[i]))
            if np.exp(-beta * (s - new_tau[i])) <= 0.001:
                break
        lambda_s = lambda_int + alpha * sum
        time_value.append(s)
        if (d*lambda_<=lambda_s):
            t=s
            tau.append(t)
    return tau

#Exponential Kernel Likelihood 
def loglikelihoodpara(para,t,integratedKernel=False):
    alpha=para[0]
    beta =para[1]
    mu=para[2]
    
    tend = t[-1]-t
    
    ll= mu*t[-1]+(alpha/beta)*len(t)
    ll2 = np.sum(-(alpha/beta)*np.exp(-beta*tend))
    ll = ll+ll2-np.log(mu)
    if integratedKernel==True:
        print("type,","integrated part,","likelihood")
        print('exp',ll+np.log(mu))
    for i in range(2,len(t)+1,1):
        li = max(i-30,0)
        temp = t[i-1]-t[li:i-1]
        
        logLam = -np.log(mu+np.sum(alpha*np.exp(-beta*temp)))
        ll = ll+logLam
    return ll


#Power Kernel Simulation 
def simu_univariate_power(alpha,delta,beta,mu,T):
    np.random.seed(seed=None)
    lambda_int=mu
    s=0
    tau=[]
    tau.append(0)
    lambda_val=[]
    time_value=[]
    time_value.append(0)
    lambda_val.append(lambda_int+alpha*np.sum(((s-tau[i]+delta)**(-beta)) for i in range(len(tau))))
    while s<T:
        sumx=0
        new_tau = tau[::-1]
        for i in range(len(new_tau)):
            sumx = sumx + (s-new_tau[i]+delta)**(-beta)
            if (s-new_tau[i]+delta)**(-beta)<=0.001:
                break
        lambda_ = lambda_int + alpha * sumx

        u=np.random.uniform(0,1,1)[0]
        w=-np.log(u)/lambda_
        s=s+w
        d=np.random.uniform(0,1,1)[0]
        sumx = 0

        for i in range(len(new_tau)):
            sumx = sumx + (s-new_tau[i]+delta)**(-beta)
            if (s-new_tau[i]+delta)**(-beta) <= 0.001:
                break
        lambda_s = lambda_int + alpha * sumx
        time_value.append(s)
        if (d*lambda_<=lambda_s):
            t=s
            tau.append(t)
    return tau


#Power Kernel Likelihood
def loglikelihood_power(para,t,integratedKernel=False):
    alpha=para[0]
    delta = para[1]
    beta =para[2]
    mu=para[3]
    
    tend = t[-1] - t
    
    expon = (-beta)+1
    num = ((delta+tend)**expon)-(delta**expon)
    ll= mu*t[-1]
    ll2 = np.sum(alpha*num/expon)
    ll = ll+ll2-np.log(mu)
    if integratedKernel==True:
        print("type - power","\n integrated part-", ll+np.log(mu))
    for i in range(2,len(t)+1,1):
        li = max(i-30,0)
        temp = t[i-1]-t[li:i-1]
        
        logLam = -np.log(mu+np.sum(alpha*((delta+temp)**(-beta))))
        ll = ll+logLam
    return ll


#Rectangular Kernel Simulation 
def simu_tick_rect(alpha,delta,beta,mu,T):
    dx=0.01
    tk = np.arange(0,6,dx)
    t_values1 = tk
    y=alpha*beta*((tk>delta)*(tk<(delta+1/beta)))
    tf1 = TimeFunction([t_values1, y],
                   inter_mode=TimeFunction.InterConstRight, dt=0.01)
    kernel1 = HawkesKernelTimeFunc(tf1)
    #kernel=HawkesKernelPowerLaw(alpha, delta, beta)#alpha=0.5,delta=1,beta=3
    hawkes = SimuHawkes(n_nodes=1, end_time=T,seed=None)

    hawkes.set_kernel(0,0,kernel1)
    hawkes.set_baseline(0, mu)
    hawkes.simulate()
    tau= hawkes.timestamps
    return tau

#Rectangular Kernel Likelihood
def loglikelihood_rect(para,t,integratedKernel=False):
    alpha=para[0]
    beta =para[1]
    delta = para[2]
    mu=para[3]
   
   
    
    tend = t[-1]-t
    
    condition1 = (tend>(delta+1/beta))
    condition2 = (tend>delta)*(tend<(delta+1/beta))
    ll= mu*t[-1]
    ll2 = np.sum(alpha*(condition1)+alpha*beta*(tend-delta)*(condition2))
    ll = ll+ll2
    llinit=ll
    ll = ll-np.log(mu)
    
   
    for i in range(2,len(t)+1,1):
        li = max(i-30,0)
        temp = t[i-1]-t[li:i-1]
        
        logLam = -np.log(mu+np.sum(alpha*beta*((temp>delta)*(temp<(delta+1/beta)))))
        ll = ll+logLam
    if integratedKernel==True:
        print("type-","rect","\n integrated part-",llinit,"\n log part-",ll-llinit)
    return ll

#Multiplicative Kernel Simulation
def simu_multiplicativeKernel(alpha,beta,mu,T,seed=None):
    np.random.seed(seed=seed)
    lambda_int=mu
    s=0
    tau=[]
    tau.append(0)
    lambda_val=[]
    time_value=[]
    time_value.append(0)
    lambda_val.append(np.exp(lambda_int))
    while s<T:
        sum=1
        new_tau = tau[::-1]
        for i in range(len(new_tau[0:20])):
            sum = (sum * kernel1(alpha,beta,s-new_tau[i]))
        lambda_ = max(np.exp(lambda_int),np.exp(lambda_int)*sum)

        u=np.random.uniform(0,1,1)[0]
        w=-np.log(u)/lambda_
        s=s+w
        d=np.random.uniform(0,1,1)[0]
        sum = 1

        for i in range(len(new_tau[0:20])):
            sum = sum*(kernel1(alpha,beta,s-new_tau[i]))
        lambda_s = np.exp(lambda_int) *sum
        time_value.append(s)
        if (d*lambda_<=lambda_s):
            t=s
            tau.append(t)
            lambda_val.append(lambda_s)
    return tau


def kernel1(alpha,beta,t):
    y=alpha*np.exp(-beta*t)
    return np.exp(y)



def kernel2(alpha,beta,t): 
    y=alpha*np.exp(-beta*t)
    return (y)


#Multiplicative Kernel Likelihood
def integratedKernel(t1,t2,temp):
    n=500
    h=(t2-t1)/n
   
    intgr=0
    intgr=intgr+h*np.sum(np.exp(mu_t+np.sum(kernel2((t1+i*h)-temp))) for i in range(0,n-1))
    return intgr
    
def loglikelihoodpara1(para,t,integratedKernel=False):
    alpha=para[0]
    beta =para[1]
    mu=para[2]
    integrated1=0
    for j in range(1,len(t)):
        tj=t[j]
        if tj>0:
            ip=t[j-1]
            ti=max(j-20,0)
            temp=t[ti:j]
        
            integrated1=integrated1+np.sum(integratedKernel(ip,tj,temp))
    ll=integrated1
    ll = ll-mu
    if integratedKernel==True:
        print("type -exp","\n integrated part-",ll)
    for i in range(2,len(t)+1,1):
        li = max(i-30,0)
        temp = t[i-1]-t[li:i-1]
        logLam = -(mu+np.sum(kernel2(alpha,beta,temp)))
        ll = ll+logLam
    return ll

#Negative Kernel Simulation
def simu_uniKernel_max(alpha,beta,mu,T):
    
    np.random.seed(seed=None)
    lambda_int=mu
    s=1
    tau=[]
    tau.append(0)
    lambda_val=[]
    time_value=[]
    time_value.append(0)
    lambda_val.append(np.exp(lambda_int))

    while s<T:
   

        sum1=0
        new_tau = tau[::-1]
        for i in range(len(new_tau[0:20])):
            sum1 = sum1+ (kernel_max(alpha,beta,s-new_tau[i]))
        lambda_ = max(mu,mu+sum1)
      

        u=np.random.uniform(0,1,1)[0]
        w=-np.log(u)/lambda_
        s=s+w
        d=np.random.uniform(0,1,1)[0]
        sum1 = 0

        for i in range(len(new_tau[0:20])):
            sum1 = sum1+(kernel_max(alpha,beta,s-new_tau[i]))
        lambda_s =  max(0,mu+sum1)
        time_value.append(s)
        if (d*lambda_<=lambda_s):
            t=s
            tau.append(t)
            lambda_val.append(lambda_s)
    return tau


def kernel_max(alpha,beta,t):
    y=alpha*np.exp(-beta*t)
    return y

#Negative Kernel Likelihood

def integratedKernel_max(mu,alpha,beta,t1,t2,temp):
    n=20
    h=(t2-t1)/n
   
    intgr=0
    intgr=intgr+h*np.sum((max(0,mu+np.sum(kernel_max(alpha,beta,(t1+i*h)-temp)))) for i in range(0,n-1))
    return intgr
    
def loglikelihoodpara_max(para,t,integratedKernel=False):
    alpha=para[0]
    beta =para[1]
    mu=para[2]
    integrated1=0
    for j in range(1,len(t)):
        tj=t[j]
        if tj>0:
            ip=t[j-1]
            ti=max(j-20,0)
            temp=t[ti:j]
            integrated1=integrated1+np.sum(integratedKernel_max(mu,alpha,beta,ip,tj,temp))
    ll=integrated1
    ll = ll-np.log(mu)
    if integratedKernel==True:
        print("type,","integrated part,","likelihood")
        print('exp',ll)
    for i in range(1,len(t)+1,1):
        li = max(i-30,0)
        temp = t[i-1]-t[li:i-1]
        if (mu+np.sum(kernel_max(alpha,beta,temp)))>0:
            logLam = -np.log(max(0,mu+np.sum(kernel_max(alpha,beta,temp))))
        else:
            logLam=0
        ll = ll+logLam
    return ll


# ## Varying Mu and Kernel Combined Simualtion and Likelihood Estimation

# In[21]:




def hawkesfunction(alpha,beta,delta,t,kernelType):
    if kernelType==0:
        kernel=alpha*np.exp(-beta*t)
    elif kernelType==1:
        kernel=alpha*beta*((t>delta)*(t<(delta+1/beta)))
    elif kernelType==5:
        kernel=(alpha-beta)*(t>=1/delta)*(t<1/alpha)+(delta-alpha)*(t>=1/alpha)*(t<1/(delta-beta))
    else:
        kernel=alpha*beta
    return kernel

        
             
def simulationVaryingMusHawkesOneD(alpha,beta,T,delta=0,muType=4,kernelType=0):
    mu_values=mufunction(np.arange(0,T,0.1),muType)
    maxLambda=max(mu_values)
    s=0
    lambdas=[]
    time_value=[]
    time_value.append(s)
    lambdas.append(mufunction(s,muType))
    
    while s<T:
        sums=0
        new_tau = time_value[::-1]
        for i in range(len(new_tau[0:30])):
            sums = sums + (hawkesfunction(alpha,beta,delta,s-new_tau[i],kernelType))
        lambda_ = max(maxLambda+sums,0)
        
        u=np.random.uniform(0,1,1)[0]
        w=-np.log(u)/lambda_
        
        s=s+w
        d=np.random.uniform(0,1,1)[0]
        sums1 = 0

        for i in range(len(new_tau[0:30])):
            sums1 = sums1 + (hawkesfunction(alpha,beta,delta,s-new_tau[i],kernelType))
        
        lambda_int=mufunction(s,muType)
        lambda_s = max(lambda_int +  sums1,0)
        if (d*lambda_<=lambda_s):
            t=s
            time_value.append(t)
            lambdas.append(lambda_s)
            
    return time_value,lambdas


def IntegratedKernelVaryingMuHawkes(alpha,beta,delta,x,muType,kernelType):
    xmax=x[-1]
    ll=0
    
    if muType==0:
        integratedMu=a*(1-np.exp(-b*xmax))/b
    elif muType==1:
        integratedMu=a/(3*c)*((c*xmax-b)**3-(c*x[0]-b)**3)
    elif muType==2:
        integratedMu=mu_t*xmax
        
    else:
        y1=d*(-np.cos(a*2*np.pi*xmax-b)/(np.pi*2*a)+c*xmax)
        y2=d*(-np.cos((a*2*np.pi*x[0]-b))/(2*a*np.pi)+c*x[0])
        integratedMu=y1-y2
    
    
    if kernelType==0:
        tend = xmax-x
        ll= integratedMu +(alpha/beta)*len(x)
        ll2 = np.sum(-(alpha/beta)*np.exp(-beta*tend))
        ll = ll+ll2
        print(ll,"exp")
    elif kernelType==1:
        tend = xmax-x
        condition1 = (tend>(delta+1/beta))
        condition2 = (tend>delta)*(tend<(delta+1/beta))
        ll2 = np.sum(alpha*(condition1)+alpha*beta*(tend-delta)*(condition2))
        ll = ll2
        print(ll,"rect")
    elif kernelType==5:
        tend=xmax-x
        condition1=(tend>=1/delta)*(tend<1/alpha)
        condition2=(tend>=1/alpha)*(tend<1/(delta-beta))
        condition3=(tend>=1/(delta-beta))
        ll2=np.sum((alpha-beta)*(tend-1/delta)*condition1+(delta-alpha)*(tend-1/alpha)*condition2+(delta-alpha)*(1/(delta-beta)-1/alpha)*condition3)
        ll = ll2
        print(ll,"new")
    return ll

    
def likelihoodVaryingMuHawkesOneD(alpha,beta,tseries,delta=0,muType=4,kernelType=0):
    ll=0
    ll2=IntegratedKernelVaryingMuHawkes(alpha,beta,delta,tseries,muType,kernelType)
    ll=ll+ll2
    print(ll)
    for i in range(2,len(tseries)+1,1):
        li = max(i-30,0)
        temp = tseries[i-1]-tseries[li:i-1]
        mu=mufunction(tseries[i-1],muType)
        
        logLam = -np.log(max(mu+np.sum(hawkesfunction(alpha,beta,delta,temp,kernelType)),1e-5))
        ll = ll+logLam
    return ll

def anyKernel(tp,p,k,types,kernels):
    if types[p][k]=='exp':
        alpha1 = kernels[p][k][0]
        beta1 = kernels[p][k][1]
        res = alpha1*np.exp(-beta1*tp)
    elif types[p][k]=='neg':
        alpha1 = kernels[p][k][0]
        beta1 = kernels[p][k][1]
        res = -alpha1*np.exp(-beta1*tp)
    elif types[p][k]=='rect':
        alpha1 = kernels[p][k][0]
        beta1 = kernels[p][k][1]
        delta1=kernels[p][k][2]
        res = alpha1*beta1*((delta1<tp)*((delta1+1/beta1)>tp))
   
    return res
def multiSimHawkesAll(mu,T,types,kernels):
    support=30
    totalD=len(mu)
    timesteps=[[] for d in range(totalD)]
    #for d in range(totalD):
    #    timesteps[d].append(0)
    tmax=0
    while tmax<T:
        lambda1=np.zeros(totalD)
        
        for d in range(totalD):
            lambda1[d]=mu[d]
            for k in range(totalD):
                if len(timesteps[k])!=0:
                    temp=tmax-np.array(timesteps[k][-support:])
                    temp=temp[temp>=0]
                    lambda1[d]+=anyKernel(temp,d,k,types,kernels).sum()        
        totalLambda=max(lambda1.sum(),0)
        u=np.random.uniform(0,1,1)[0]
        intArr=-np.log(u)/totalLambda
        
        tmax=tmax+intArr
        u=np.random.uniform(0,1,1)[0]
        lambdaTmax=np.zeros(totalD)
        for d in range(totalD):
            lambdaTmax[d]=mu[d]
            for k in range(totalD):
                if len(timesteps[k])!=0:
                    temp=tmax-np.array(timesteps[k][-support:])
                    temp=temp[temp>0]
                    lambdaTmax[d]+=anyKernel(temp,d,k,types,kernels).sum()
            if totalLambda*u<=max(lambdaTmax.sum(),0):
                timesteps[d].append(tmax)
                break
    return timesteps

dictdimP={}
mapping={}
def definitions(t):
    totalD=len(t)
    
    for j in range(totalD):
        if t[j][0]!=0:
            t[j]=np.insert(t[j],0,0)
            
    dimensions=np.arange(0,totalD,1)
    for i in range(totalD):
        dictdimP[i]=np.delete(dimensions,i)

    for i in range(totalD):
        mapping[i,i]=createMapAtoBIndex(t[i],t[i])
        for j in (dictdimP[i]):
            mapping[i,j]=createMapAtoBIndex(t[i],t[j])
    tmax=0
    for p in range(totalD):
        tmax=max(tmax,t[p][-1])
    return tmax
    
def loglikelihoodMulti(t,mu,types,kernels):
    tmax = definitions(t)
    totalD=len(t)
    for j in range(totalD):
        if t[j][0]!=0:
            t[j]=np.insert(t[j],0,0)
  
    ll=0
    integrated=0
    for p in range(0,totalD,1):
        a=0
        if types[p][p]=='neg':
            tend=t[p]
        else:    
            tend = (t[p][-1]-t[p][:])
            a+=mu[p]*(t[p][-1]-t[p][0])
        a+=np.sum(anyRandomIntegratedKernel(tend.reshape(1,-1),mu,p,p,types,kernels))
    
        for k in dictdimP[p]:
            if types[p][p]=='neg':
                tend=t[p]
            else:
                tend=(t[p][-1]-t[k][:])
                tend=tend[tend>=0]
            a+=np.sum(anyRandomIntegratedKernel(tend.reshape(1,-1),mu,p,k,types,kernels))
        
        integrated+=a

        ll = ll+a
        
        ll = ll-np.log(mu[p])
       
        tp = t[p][:]  
        
        for i in range(1,len(tp),1):
            li = max(i-30,0)
            temp1 = tp[i]-tp[li:i]
            decayFactor = np.sum(anyKernel(temp1.reshape(1,-1),p,p,types,kernels))
            for k in dictdimP[p]:
                jT = mapping[p,k].get(tp[i])
                if(jT != None):
                    j = jT
                    lj = max(j-30,0)
                    temp2 = tp[i]-t[k][lj:j+1]
                    decayFactor = decayFactor + np.sum(anyKernel(temp2.reshape(1,-1),p,k,types,kernels))
            logLam = -np.log(max(mu[p]+decayFactor,1e-5))
            ll = ll+logLam
    print("Integrated Kernel",integrated)
        
    return ll

def anyRandomIntegratedKernel(tend,mu,p,k,types,kernels):
    if types[p][k]=='exp':
        alpha1 = kernels[p][k][0]
        beta1 = kernels[p][k][1]
        res = (alpha1/beta1)*(1-np.exp(-beta1*tend))
    elif types[p][k]=='rect':
        alpha1 = kernels[p][k][0]
        beta1 = kernels[p][k][1]
        delta1=kernels[p][k][2]
        condition1 = (tend>(delta1+1/beta1))
        condition2 = (tend>delta1)*(tend<(delta1+1/beta1))
        ll2 = np.sum(alpha1*(condition1)+alpha1*beta1*(tend-delta1)*(condition2))
        res=ll2
    elif types[p][k]=='neg':
        alpha1 = kernels[p][k][0]
        beta1 = kernels[p][k][1]
        res=0
        for j in range(1,len(tend)):
            
            tj=tend[j]
            if tj>0:
                ip=tend[j-1]
                ti=max(j-30,0)
                temp=tend[ti:j]
                n=20
                h=(ti-ip)/n
                intgr=0
                intgr=intgr+h*np.sum((max(0,np.sum(-alpha*np.exp(-beta*(ip+i*h)-temp)))) for i in range(0,n-1))
            
            res=res+intgr
    
    return res

def anyKernel(tp,p,k,types,kernels):
    if types[p][k]=='exp':
        alpha1 = kernels[p][k][0]
        beta1 = kernels[p][k][1]
        res = alpha1*np.exp(-beta1*tp)
    elif types[p][k]=='neg':
        alpha1 = kernels[p][k][0]
        beta1 = kernels[p][k][1]
        res = -alpha1*np.exp(-beta1*tp)
    elif types[p][k]=='rect':
        alpha1 = kernels[p][k][0]
        beta1 = kernels[p][k][1]
        delta1=kernels[p][k][2]
        res = alpha1*beta1*((delta1<tp)*((delta1+1/beta1)>tp))
   
    return res


def createMapAtoBIndex(a,b):
    mapAtoBIndex={}
    for x in a:
        if(max(b[b<x],default=-1)==-1):
            mapAtoBIndex[x] = None
        else:
            mapAtoBIndex[x] = (np.where(b==max(b[b<x])))[0][0]
    return mapAtoBIndex

