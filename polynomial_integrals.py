import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def build_dictionary(param_ranges,max_degree):
    no_variables = len(param_ranges)
    dict_integrals = {}
    param_limits = param_ranges
    for i in range(no_variables):
        substr = 'x' + str(i)
        for j in range(max_degree+1):
            denominator = j + 1.0
            power = j + 1
            numerator = np.power(param_limits[i][1], power) - np.power(param_limits[i][0], power)
            coeff = (numerator) / (denominator)
            if j == 0:
                subsubstr = substr + '_const'
            elif j == 1:
                subsubstr = substr
            else:
                subsubstr = substr + '^' + str(j)
            dict_integrals[subsubstr] = coeff
    return dict_integrals

def calculate_integral_multipliers(param_ranges,max_degree):
    p=PolynomialFeatures(max_degree)
    p.fit_transform(np.random.random((1,len(param_ranges)))) #fitting a random value belonging to the param space

    featureNames = p.get_feature_names()

    nam = featureNames
    print(nam)
    list_coeff = np.ones((len(nam),len(param_ranges)))
    dict_integrals = build_dictionary(param_ranges,max_degree)
    no_variables = len(param_ranges)

    for expr, ite in zip(nam, range(len(nam))):
        variables = expr.split(' ')
        for variable in variables:
            for i in range(no_variables):
                variable_str = 'x' + str(i)
                if variable_str in variable:
                    list_coeff[ite, i]*= dict_integrals[variable]
                else:
                    list_coeff[ite, i]*= dict_integrals[variable_str + '_const']
    return list_coeff
print(calculate_integral_multipliers([[0,1],[1,2],[2,3]],2))


# param_limits = [[1,2],[2,3]]
# max_degree = 2
# calculate_multipliers(param_limits,max_degree)


