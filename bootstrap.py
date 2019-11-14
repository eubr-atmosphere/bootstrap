import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, Lasso

class Bootstrap():
    def __init__(self, filename):
        df = pd.read_csv(filename)
        
        self.categorical_features = df.columns[[2,5,6]]
        self.features_to_drop = df.columns[[0,1,3,4,15,16,17,18]]
        self.y_name = df.columns[14]
        
        df = self.__preprocess__(df)
        self.y = df[self.y_name]
        self.X = df.drop(columns=self.y_name) 
        
        self.reg, self.error = self.__fitAll__()
        self.models = self.__getBootstrapModels__()
        
    def __getTrainingData__(self):
        return self.X, self.y

    def __fitAll__(self):
        reg = LassoCV(cv=5, n_alphas=10, random_state=1).fit(self.X, self.y)
        error = reg.predict(self.X)-self.y.to_numpy()
        
        # print('Average y', self.y.mean())
        # print('RMSE:',np.sqrt(np.average(error**2)))
        # print('MAPE:', np.average(np.abs(error)/self.y.to_numpy()) )
        # print('Fraction error above 30%:', np.average(np.abs(error)/self.y.to_numpy()>0.3) )
        
        return reg, error

    def __fit__(self,X,y):
        return Lasso(alpha=self.reg.alpha_).fit(X, y)
    
    def __getBootstrapModels__(self):
        # Array of responses
        n = len(self.y)
        size = int(np.ceil(np.sqrt(n)))       
        models = []
        # Sample (size n with replacement)
        for j in range(size):
            inds = np.random.choice(self.X.index, size=n)
            models.append(self.__fit__(self.X.loc[inds],self.y.loc[inds]))
            
        return models

        
    def __preprocess__(self,df):
        categorical_features = list(set(df.columns) & set(self.categorical_features))
        features_to_drop = list(set(df.columns) & set(self.features_to_drop))
        for feature in categorical_features:
            dummy = pd.get_dummies(df[feature])
            df = pd.concat([df,dummy],axis=1)

        df = df.drop(columns = list(categorical_features)+list(features_to_drop))
        return df
    
    
    def getBootstrapCIs(self, alpha, x):
        """
        Bootstrap-based confidence intervals algorithm
        :param alpha: (1-alpha) is the confidence level
        :param x: feature vector for which CIs must be computed
        :return: Bootstrap-based confidence interval
        """
        x = self.__preprocess__(x)
        
        # Sample (size n with replacement)
        M = np.zeros(len(self.models))
        for j, model in enumerate(self.models):
            M[j] = model.predict(x)
        M -= M.mean()

        # Error sample set
        C = set()
        for e in self.error:
            for mj in M:
                C.add(e + mj)
        C = np.array(list(C))
        
        return self.reg.predict(x) + np.percentile(C, [(alpha/2)*100, (1 - alpha/2)*100])
