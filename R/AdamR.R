# Minimize an arbitrary function using Adaptive Moment estimation (Adam)
#
# @param theta initial values for the parameters to be optimized over.
# @param f  a function to be minimized
# @param data a matrix which contains all other information besides theta in f
# @param batch.size number of observations used each time
# @param alpha learning rate
# @param beta1 exponential decay rates for the 1st moment estimates
# @param beta2 exponential decay rates for the 2nd moment estimates
# @param thres stopping threshold
# @param maxepoch maximum number of epochs
# This function returns the best set of parameters and the corresponding value of f
AdamR=function(theta,f,data,batch.size=100,alpha=0.1,beta1=0.9,beta2=0.999,thres=1e-3,maxepoch=10000){
  data=as.matrix(data)

  # Initialize 1st and 2nd moment vector
  m = rep(0,length(theta))
  v = rep(0,length(theta))

  # Epsilon
  epsilon=1e-8

  # Timestep
  t=0

  # Epoch
  epoch=0

  # Batch management
  k=floor(nrow(data)/batch.size)
  rem = nrow(data)%%batch.size

  # Start
  repeat{
    subset = (1-batch.size):0
    for(i in 1:(k-1)){
      # Subset the data
      subset=subset+batch.size
      feed<<-data[subset,]

      # Adam
      t=t+1
      alphat=alpha/sqrt(t)
      g = numDeriv::jacobian(f,theta)
      m = beta1*m + (1-beta1)*g
      v = beta2*v + (1-beta2)*g^2
      mhat = m/(1-beta1^t)
      vhat = v/(1-beta2^t)
      new.theta=theta - as.vector(alphat*mhat/(sqrt(vhat)+epsilon))
      if(max(abs(new.theta-theta))<thres){return(list(best.theta = new.theta,value = f(new.theta)))}
      theta=new.theta
    }

    # Run the last batch
    subset=(subset[batch.size]+1):nrow(data)
    feed<<-data[subset,]

    # Adam
    t=t+1
    alphat=alpha/sqrt(t)
    g = jacobian(f,theta)
    m = beta1*m + (1-beta1)*g
    v = beta2*v + (1-beta2)*g^2
    mhat = m/(1-beta1^t)
    vhat = v/(1-beta2^t)
    new.theta=theta - as.vector(alphat*mhat/(sqrt(vhat)+epsilon))
    if(max(abs(new.theta-theta))<thres){return(list(best.theta = new.theta,value = f(new.theta)))}
    theta=new.theta

    # Count epoch
    epoch = epoch + 1
    if(epoch>maxepoch){return(list(best.theta = new.theta,value = f(new.theta)))}
  }
}
