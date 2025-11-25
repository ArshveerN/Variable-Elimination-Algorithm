from bnetbase import Variable, Factor, BN
import csv
import itertools


def normalize(factor):
    '''
    Normalize the factor such that its values sum to 1.
    Do not modify the input factor.

    :param factor: a Factor object.
    :return: a new Factor object resulting from normalizing factor.
    '''
    totalSum = 0
    for value in factor.values:
        totalSum += value
    if totalSum == 0:
        return None

    newFactor = Factor(factor.name, factor.get_scope())

    for index, value in enumerate(factor.values):
        newFactor.values[index] = value / totalSum

    return newFactor


def restrict(factor, variable, value):
    '''
    Restrict a factor by assigning value to variable.
    Do not modify the input factor.

    :param factor: a Factor object.
    :param variable: the variable to restrict.
    :param value: the value to restrict the variable to
    :return: a new Factor object resulting from restricting variable to value.
             This new factor no longer has variable in it.

    '''
    # Your code here!
    return None


def sum_out(factor, variable):
    '''
    Sum out a variable variable from factor factor.
    Do not modify the input factor.

    :param factor: a Factor object.
    :param variable: the variable to sum out.
    :return: a new Factor object resulting from summing out variable from the factor.
             This new factor no longer has variable in it.
    '''
    # Your code here!
    return None


def multiply(factor_list):
    '''
    Multiply a list of factors together.
    Do not modify any of the input factors.

    :param factor_list: a list of Factor objects.
    :return: a new Factor object resulting from multiplying all the factors in factor_list.
    '''
    # Your code here!
    return None

def ve(bayes_net, var_query, EvidenceVars):
    '''

    Execute the variable elimination algorithm on the Bayesian network bayes_net
    to compute a distribution over the values of var_query given the
    evidence provided by EvidenceVars.

    :param bayes_net: a BN object.
    :param var_query: the query variable. we want to compute a distribution
                     over the values of the query variable.
    :param EvidenceVars: the evidence variables. Each evidence variable has
                         its evidence set to a value from its domain
                         using set_evidence.
    :return: a Factor object representing a distribution over the values
             of var_query. that is a list of numbers, one for every value
             in var_query's domain. These numbers sum to 1. The i-th number
             is the probability that var_query is equal to its i-th value given
             the settings of the evidence variables.

    For example, assume that
        var_query = A with Dom[A] = ['a', 'b', 'c'],
        EvidenceVars = [B, C], and
        we have called B.set_evidence(1) and C.set_evidence('c'),
    then VE would return a list of three numbers, e.g. [0.5, 0.24, 0.26].
    These numbers would mean that
        Pr(A='a'|B=1, C='c') = 0.5,
        Pr(A='a'|B=1, C='c') = 0.24, and
        Pr(A='a'|B=1, C='c') = 0.26.

    '''
    # Your code here!
    return None


def naive_bayes_model(data_file, variable_domains = {"Work": ['Not Working', 'Government', 'Private', 'Self-emp'], "Education": ['<Gr12', 'HS-Graduate', 'Associate', 'Professional', 'Bachelors', 'Masters', 'Doctorate'], "Occupation": ['Admin', 'Military', 'Manual Labour', 'Office Labour', 'Service', 'Professional'], "MaritalStatus": ['Not-Married', 'Married', 'Separated', 'Widowed'], "Relationship": ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'], "Race": ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'], "Gender": ['Male', 'Female'], "Country": ['North-America', 'South-America', 'Europe', 'Asia', 'Middle-East', 'Carribean'], "Salary": ['<50K', '>=50K']}, class_var = Variable("Salary", ['<50K', '>=50K'])):
    '''
    NaiveBayesModel returns a BN that is a Naive Bayes model that
    represents the joint distribution of value assignments to
    variables in the Adult Dataset from UCI.  Remember a Naive Bayes model
    assumes P(X1, X2,.... XN, Class) can be represented as
    P(X1|Class)*P(X2|Class)* .... *P(XN|Class)*P(Class).
    When you generated your Bayes bayes_net, assume that the values
    in the SALARY column of the dataset are the CLASS that we want to predict.
    @return a BN that is a Naive Bayes model and which represents the Adult Dataset.
    '''
    ### READ IN THE DATA
    input_data = []
    with open(data_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None) #skip header row
        for row in reader:
            input_data.append(row)

    ### DOMAIN INFORMATION REFLECTS ORDER OF COLUMNS IN THE DATA SET
    #variable_domains = {
    #"Work": ['Not Working', 'Government', 'Private', 'Self-emp'],
    #"Education": ['<Gr12', 'HS-Graduate', 'Associate', 'Professional', 'Bachelors', 'Masters', 'Doctorate'],
    #"Occupation": ['Admin', 'Military', 'Manual Labour', 'Office Labour', 'Service', 'Professional'],
    #"MaritalStatus": ['Not-Married', 'Married', 'Separated', 'Widowed'],
    #"Relationship": ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
    #"Race": ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
    #"Gender": ['Male', 'Female'],
    #"Country": ['North-America', 'South-America', 'Europe', 'Asia', 'Middle-East', 'Carribean'],
    #"Salary": ['<50K', '>=50K']
    #}

    # Your code here!
    return None

def explore(bayes_net, question):
    '''    Input: bayes_net---a BN object (a Bayes bayes_net)
           question---an integer indicating the question in HW4 to be calculated. Options are:
           1. What percentage of the women in the data set end up with a P(S=">=$50K"|E1) that is strictly greater than P(S=">=$50K"|E2)?
           2. What percentage of the men in the data set end up with a P(S=">=$50K"|E1) that is strictly greater than P(S=">=$50K"|E2)?
           3. What percentage of the women in the data set with P(S=">=$50K"|E1) > 0.5 actually have a salary over $50K?
           4. What percentage of the men in the data set with P(S=">=$50K"|E1) > 0.5 actually have a salary over $50K?
           5. What percentage of the women in the data set are assigned a P(Salary=">=$50K"|E1) > 0.5, overall?
           6. What percentage of the men in the data set are assigned a P(Salary=">=$50K"|E1) > 0.5, overall?
           @return a percentage (between 0 and 100)
    '''
    input_data = []
    with open('data/adult-test.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None) #skip header row
        for row in reader:
            input_data.append(row)

    # Your code here!
    return None

if __name__ == '__main__':
    nb = naive_bayes_model('data/adult-train.csv')
    for i in range(1,7):
        print("explore(nb,{}) = {}".format(i, explore(nb, i)))
