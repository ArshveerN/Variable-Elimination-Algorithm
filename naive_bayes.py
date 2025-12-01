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
    new_scope = []
    variable.set_assignment(value)

    for var in factor.get_scope():
        if var != variable:
            new_scope.append(var)


    dom = []
    for var in new_scope:
        dom.append(var.domain())

    new_factor = Factor(factor.name + f" {variable} = {value}", new_scope)

    for value_combo in itertools.product(*dom):
        for index, var in enumerate(new_scope):
            var.set_assignment(value_combo[index])

        val = factor.get_value_at_current_assignments()
        new_factor.add_value_at_current_assignment(val)

    return new_factor


def sum_out(factor, variable):
    '''
    Sum out a variable variable from factor factor.
    Do not modify the input factor.

    :param factor: a Factor object.
    :param variable: the variable to sum out.
    :return: a new Factor object resulting from summing out variable from the factor.
             This new factor no longer has variable in it.
    '''
    new_scope = []
    for v in factor.get_scope():
        if v != variable:
            new_scope.append(v)

    new_factor = Factor(factor.name + f" {variable} Summed out", new_scope)

    domains = []
    for v in new_scope:
        domains.append(v.domain())

    for value_combo in itertools.product(*domains):
        for index, value in enumerate(new_scope):
            value.set_assignment(value_combo[index])
        sum_ = 0
        for val in variable.domain():
            variable.set_assignment(val)
            sum_ += factor.get_value_at_current_assignments()

        new_factor.add_value_at_current_assignment(sum_)

    return new_factor

def multiply(factor_list):
    '''
    Multiply a list of factors together.
    Do not modify any of the input factors.

    :param factor_list: a list of Factor objects.
    :return: a new Factor object resulting from multiplying all the factors in factor_list.
    '''
    if len(factor_list) == 0:
        return None
    scope = []
    name = ""
    for factor in factor_list:
        name += factor.name
        for variable in factor.scope:
            if variable not in scope:
                scope.append(variable)

    new_factor = Factor("Multiplied " + name, scope)
    domains = []
    for v in scope:
        domains.append(v.domain())

    for value_combo in itertools.product(*domains):
        for index, value in enumerate(scope):
            value.set_assignment(value_combo[index])
        mult_ = 1
        for factor in factor_list:
            mult_ *= factor.get_value_at_current_assignments()

        new_factor.add_value_at_current_assignment(mult_)

    return new_factor



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
    factors = list(bayes_net.factors())
    factors = restrict_factors_by_evidence(factors, EvidenceVars)
    factors = eliminate_variables(factors, bayes_net, var_query, EvidenceVars)
    final_factor = multiply(factors)
    normalized = normalize(final_factor)
    return normalized


def eliminate_variables(factors, bayes_net, var_query, EvidenceVars):
    vars_to_eliminate = []
    all_vars = bayes_net.variables()
    for var in all_vars:
        if var == var_query:
            continue
        if var in EvidenceVars:
            continue
        vars_to_eliminate.append(var)

    for var in vars_to_eliminate:
        with_var = []
        without_var = []

        for factor in factors:
            if var in factor.get_scope():
                with_var.append(factor)
            else:
                without_var.append(factor)

        if len(with_var) > 0:
            combined = multiply(with_var)
            summed_out = sum_out(combined, var)
            factors = without_var
            factors.append(summed_out)
    return factors


def restrict_factors_by_evidence(factors, EvidenceVars):
    current_factors = factors
    for evidence_var in EvidenceVars:
        evidence_value = evidence_var.get_evidence()
        new_factors = []
        for factor in current_factors:
            if evidence_var in factor.get_scope():
                new_factors.append(restrict(factor, evidence_var, evidence_value))
            else:
                new_factors.append(factor)
        current_factors = new_factors
    return current_factors


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
    variable_domains = {
    "Work": ['Not Working', 'Government', 'Private', 'Self-emp'],
    "Education": ['<Gr12', 'HS-Graduate', 'Associate', 'Professional', 'Bachelors', 'Masters', 'Doctorate'],
    "Occupation": ['Admin', 'Military', 'Manual Labour', 'Office Labour', 'Service', 'Professional'],
    "MaritalStatus": ['Not-Married', 'Married', 'Separated', 'Widowed'],
    "Relationship": ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
    "Race": ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
    "Gender": ['Male', 'Female'],
    "Country": ['North-America', 'South-America', 'Europe', 'Asia', 'Middle-East', 'Carribean'],
    "Salary": ['<50K', '>=50K']
    }
    all_vars = []
    for attr_name in variable_domains.keys():
        var = Variable(attr_name, variable_domains[attr_name])
        all_vars.append(var)

    all_variables = [class_var] + all_vars

    class_factor = Factor("Salary", [class_var])
    all_factors = [class_factor]

    for var in all_vars:
        factor = Factor(f"{var.name}", [var, class_var])
        all_factors.append(factor)
    
    counts = {"Salary": {}}
    for salary_val in class_var.domain():
        counts["Salary"][salary_val] = 0

    for var in all_vars:
        counts[var.name] = {}
        for attr_val in var.domain():
            for salary_val in class_var.domain():
                counts[var.name][(attr_val, salary_val)] = 0

    for row in input_data:
        salary_val = row[8]
        counts["Salary"][salary_val] += 1

        for i, var in enumerate(all_vars):
            attr_val = row[i]
            counts[var.name][(attr_val, salary_val)] += 1

    total = len(input_data)

    salary_probs = []
    for salary_val in class_var.domain():
        prob = counts["Salary"][salary_val] / total
        salary_probs.append([salary_val, prob])
    class_factor.add_values(salary_probs)

    for idx, var in enumerate(all_vars):
        attr_probs = []
        for attr_val in var.domain():
            for salary_val in class_var.domain():
                joint_count = counts[var.name][(attr_val, salary_val)]
                salary_count = counts["Salary"][salary_val]
                if salary_count > 0:
                    prob = joint_count / salary_count
                else:
                    prob = 0
                attr_probs.append([attr_val, salary_val, prob])
        all_factors[idx + 1].add_values(attr_probs)

    # Create BN
    bn = BN("Naive Bayes", all_variables, all_factors)
    return bn

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
