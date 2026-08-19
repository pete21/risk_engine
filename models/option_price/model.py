import math 
import scipy.stats as si

from models import utils

from . import api_pb2


decode_input = utils.ProtoDecoder(api_pb2.ModelInput)

def calculate(parameters, data):
    result = euro_vanilla_dividend(data.S, data.K, data.T, data.r, data.q, data.sigma, data.option)
    return {'price': result}

def euro_vanilla_dividend(S, K, T, r, q, sigma, option = 'call'):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #q: rate of continuous dividend paying asset 
    #sigma: volatility of underlying asset
    
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = (math.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    
    if option == 'call':
        result = (S * math.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) - K * math.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    if option == 'put':
        result = (K * math.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * math.exp(-q * T) * si.norm.cdf(-d1, 0.0, 1.0))
        
    return result
