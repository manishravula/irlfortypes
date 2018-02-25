#Helper functions to help with sanity checks. Need to be called according to intution. NOT A test-suite to run everytime a change is made.

import numpy as np
import matplotlib.pyplot as plt
from src.arena import arena
from src.agents.agent_originaltypes import Agent
import numpy as np
import copy
import time
import numpy.polynomial.polynomial as poly
from matplotlib.animation import FuncAnimation


def test_for_normalization(polycoeffs,xrange):
    '''
    Test to check if a polynomial integrates to one with a small error.
    :param polycoeffs: single polynomial coeffecient array or list of polynomial coefficients.
    :param xrange: the range of x-axis to integrate.
    :return: Nothing, raises exceptions in case there is no element to go with.
    '''
    polycoeffs_list = polycoeffs.tolist()
    epsilon = .001
    if isinstance(polycoeffs_list,list):
        ele = polycoeffs_list[0]
        if isinstance(ele,list):
            #list of polynomials.
            results = []
            for polynomial in polycoeffs:
                integral = poly.polyint(polynomial)
                sum = np.diff(poly.polyval(xrange,integral))[0]
                diff_from_one = np.abs(sum-1)
                if np.abs(sum-1) > epsilon:
                    results.append([False,diff_from_one])
                else:
                    results.append([True,diff_from_one])
            assert False not in results, 'Not normalized polynomial at indices {}'.format(results.index(False))
        else:
            integral = poly.polyint(polycoeffs)
            sum = np.diff(poly.polyval(xrange,integral))[0]
            diff_from_one = np.abs(sum - 1)

            # ndivs = 100000
            # x = np.linspace(xrange[0],xrange[1],ndivs)
            # sum2 = np.sum(poly.polyval(x,polycoeffs))*(xrange[1]-xrange[0])/ndivs

            # diff_from_one_2 = np.abs(sum2-1)
            # assert diff_from_one_2<epsilon, 'Not normalized polynomial with error 2 {}, {}'.format(diff_from_one_2, diff_from_one)
            assert diff_from_one<epsilon, 'Not normalized polynomial with error {} and actual sum {}'.format(diff_from_one, sum)
    return


def polynomialIntegrationFunc_test():
    n_poly = 100

    def polynomial_normalize(polycoeffs,xrange):
        '''

        :param polycoeffs: polynomial co-efficients to normalize. (according to numpy.polynomial.polynomial.polynomial convention)
        :param xrange: The [beginning, ending] point to evaluate around on the normalization on x-axis
        :return: normailzed polynomial and the normalization scale.
        '''
        integral = poly.polyint(polycoeffs)
        sum = np.diff(poly.polyval(xrange,integral))/1.0

        if not np.isnan(sum):
            normalized_polynomial = polycoeffs/sum
        # print(sum)
        # polynomial_normalize(normalized_polynomial,xrange)
            test_for_normalization(normalized_polynomial,xrange)
            return normalized_polynomial, sum


    for i in range(1,20):
        for j in range(n_poly):
            print(j)
            curr_poly = np.random.randint(-100,100,i)
            polynomial_normalize(curr_poly,[-10,10])


if __name__ == '__main__':
    polynomialIntegrationFunc_test()