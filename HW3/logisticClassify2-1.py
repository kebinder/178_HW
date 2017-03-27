import numpy as np
import mltools as ml
import matplotlib.pyplot as plt

# Fix the required "not implemented" functions for the homework ("TODO")

################################################################################
## LOGISTIC REGRESSION BINARY CLASSIFIER #######################################
################################################################################


class logisticClassify2(ml.classifier):
    """A binary (2-class) logistic regression classifier

    Attributes:
        classes : a list of the possible class labels
        theta   : linear parameters of the classifier 
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor for logisticClassify2 object.  

        Parameters: Same as "train" function; calls "train" if available

        Properties:
           classes : list of identifiers for each class
           theta   : linear coefficients of the classifier; numpy array 
        """
        self.classes = [0,1]              # (default to 0/1; replace during training)
        self.theta = np.array([])         # placeholder value before training

        if len(args) or len(kwargs):      # if we were given optional arguments,
            self.train(*args,**kwargs)    #  just pass them through to "train"


## METHODS ################################################################

    def plotBoundary(self,X,Y):
        """ Plot the (linear) decision boundary of the classifier, along with data """
        if len(self.theta) != 3: raise ValueError('Data & model must be 2D');
        ax = X.min(0),X.max(0); ax = (ax[0][0],ax[1][0],ax[0][1],ax[1][1]);
		
        ## TODO: find points on decision boundary defined by theta0 + theta1 X1 + theta2 X2 == 0
		
        x1b = np.array([ax[0],ax[1]])  # at X1 = points in x1b
        x2b = []
        for each_b_value in x1b:
            x2b.append((self.theta[0] + self.theta[1]*each_b_value)/-self.theta[2]) #finding x2

	#make x2b consistent with x1b
        x2b = np.array(x2b)
	## TODO find x2 values as a function of x1's values
        ## Now plot the data and the resulting boundary:
        A = Y==self.classes[0];                                         
	## and plot it:
        plt.plot(X[A,0],X[A,1],'b.',X[-A,0],X[-A,1],'r.',x1b,x2b,'k-'); plt.axis(ax); plt.draw();

    def predictSoft(self, X):
        """ Return the probability of each class under logistic regression """
        raise NotImplementedError
        ## You do not need to implement this function.
        ## If you *want* to, it should return an Mx2 numpy array "P", with 
        ## P[:,1] = probability of class 1 = sigma( theta*X )
        ## P[:,0] = 1 - P[:,1] = probability of class 0 
        return P

    def predict(self, X):
        """ Return the predictied class of each data point in X"""
        ## TODO: compute linear response r[i] = theta0 + theta1 X[i,1] + theta2 X[i,2] + ... for each i
        ## TODO: if z[i] > 0, predict class 1:  Yhat[i] = self.classes[1]
        ##       else predict class 0:  Yhat[i] = self.classes[0]
        X_coord, Y_coord = X.shape
        X_horizontal_stack = np.hstack((np.ones((X_coord,1)), X))
        matrix_multiplication = np.dot(self.theta, np.transpose(X_horizontal_stack))
        Yhat = matrix_multiplication > 0 #iterates through the entire matrix, making it true or false (1 or 0)
        return Yhat

    def sig(self, n):
        return 1 / (1.0 + np.exp(-n))

    def train(self, X, Y, initStep=1.0, stopTol=1e-4, stopEpochs=5000, plot=None):
        """ Train the logistic regression using stochastic gradient descent """
        M,N = X.shape;                     # initialize the model if necessary:
        self.classes = np.unique(Y);       # Y may have two classes, any values
        XX = np.hstack((np.ones((M,1)),X));   # XX is X, but with an extra column of ones
        YY = ml.toIndex(Y,self.classes);   # YY is Y, but with canonical values 0 or 1
        if len(self.theta)!=N+1: self.theta=np.random.rand(N+1);
        # init loop variables:
        epoch=0; done=False; Jnll=[]; J01=[];
        while not done:
            stepsize, epoch = initStep*2.0/(2.0+epoch), epoch+1; # update stepsize
            # Do an SGD pass through the entire data set:
            for i in np.random.permutation(M):
                ri    = sum([self.theta[j]* XX[i][j] for j in range(N+1)])#np.dot(XX[i], (self.theta))#NotImplementedError;     # TODO: compute linear response r(x)
                sig_2 = self.sig(ri)
                gradi = []#-YY[i] * (1-sig(r)) + (1-YY[i]*sig(r))#NotImplementedError;     # TODO: compute gradient of NLL loss
                for j in range(N+1):
                    if YY[i] == 1:
                        dji = -(1-sig_2) * XX[i][j]
                        gradi.append(dji)
                    else:
                        dji = (sig_2) * XX[i][j]
                        gradi.append(dji)
                gradi = np.array(gradi)
                self.theta -= stepsize * gradi;  # take a gradient step
            J01.append( self.err(X,Y) )  # evaluate the current error rate 

            ## TODO: compute surrogate loss (logistic negative log-likelihood)
            #Jnll = sum_i [ (log (si) if yi==1 else (log(1-si))) ] #surrogate loss, plug in xx at i, yy at i, gives you ri
            #append Jsur in Jnll
            #sum all the n01 function and then divide it by n

            Jsur = 0
            for i in range(M):
                ri = sum([self.theta[j] * XX[i][j] for j in range(N + 1)])
                sig_2 = self.sig(ri)
                if YY[i] == 1:
                    Jsur += -np.log(sig_2)
                else:
                    Jsur += -np.log(1.0 - sig_2)
            Jnll.append(Jsur/M)
            '''
            Jsur = np.mean(-1.* y * np.log(Jnll) - (1-y)*np.log(Jnll))
            Jnll = sum_i [ (log (si) if yi==1 else (log(1-si))) ]

            np.mean(-1.*y * np.log(si) - (1-y)*np.log(si))

            in which si == a random array
            
            for each y in something:
                if y == 1:
                    -log Pr[y=1]
                else:
                    -log Pr[y=0]

            add 2 theta to the gradient
            '''
            # TODO evaluate the current NLL loss
            plt.figure(1); plt.plot(Jnll,'b-',J01,'r-'); plt.draw();    # plot losses
            if N==2: plt.figure(2); self.plotBoundary(X,Y); plt.draw(); # & predictor if 2D
            plt.pause(.01);                    # let OS draw the plot

            ## For debugging: you may want to print current parameters & losses
            # print self.theta, ' => ', Jsur[-1], ' / ', J01[-1]  
            # input()   # pause for keystroke

            # TODO check stopping criteria: exit if exceeded # of epochs ( > stopEpochs)
            if(epoch>stopEpochs) or (len(Jnll)>2 and abs(Jnll[-1]-Jnll[-2]) < stopTol):
                done = True   # or if Jnll not changing between epochs ( < stopTol )
            done = epoch > stopEpochs
		


################################################################################
################################################################################
################################################################################

