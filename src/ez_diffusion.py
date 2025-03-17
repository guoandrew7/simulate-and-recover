import numpy as np

class EZDiffusionSimulator:
    def __init__(self, ):
        self.results = []

    def generate_true_parameters(self):
        #generate random parameters within  given ranges
        a = np.random.uniform(0.5, 2.0)
        v = np.random.uniform(0.5, 2.0)
        t = np.random.uniform(0.1, 0.5)
        return a, v, t

    def generate_predicted_summary_statistics(self,a,v,t):
        #use first 3 equations to generate
        y = np.exp(-a*v)
        r_pred = 1/(y+1)
        m_pred = t + (a/(2*v))*((1-y)/(1+y))
        v_pred = (a/(2*v**3))*((1-2*a*v*y-y**2)/(y+1)**2)
        return r_pred, m_pred, v_pred

    def compute_estimated_parameters(self,n,r_pred,m_pred,v_pred):
        #use last 3 equations to simulate observed summary statistics
        r_obs = np.random.binomial(n,r_pred)/n
        m_obs = np.random.normal(m_pred,np.sqrt(v_pred))
        v_obs = np.random.gamma((n-1)/2,(2*v_pred)/(n-1))
        
        #use observed summary statistics with middle 3 equations to compute estimated parameters
        if(r_obs == 1): #divide by 0 error
            r_obs = 0.999
        l = np.log(r_obs/(1-r_obs))
        
        v_est = np.sign(r_obs-0.5)*np.sqrt(np.sqrt((l*(r_obs**2 * l-r_obs*l+r_obs-0.5))/v_obs))
        if(v_est == 0):
            v_est = 0.001
        a_est = l/v_est
        t_est = m_obs-(a_est/(2*v_est)*(1-np.exp(-v_est*a_est))/(1+np.exp(-v_est*a_est)))
        return a_est, v_est, t_est
    
    def calculate_bias(self,n):
        b = []
        b_est = []
        for i in range(1000):
            a,v,t = self.generate_true_parameters()
            r_pred,m_pred,v_pred = self.generate_predicted_summary_statistics(a,v,t)
            predicted = self.compute_estimated_parameters(n,r_pred,m_pred,v_pred)
            b.append(tuple([a,v,t]))
            b_est.append(predicted)
        b = (np.mean(b, axis = 0) - np.mean(b_est, axis = 0))
        b_squared = b**2
        return b,b_squared
            
if __name__ == "__main.sh__":
    simulator = EZDiffusionSimulator()
    n = [10,40,4000]
    for n in n:
        b,b_squared = simulator.calculate_bias(n)
        print('When n = ' + str(n) + ', bias = ' + str(b) + ' and squared bias = ' + str(b_squared))