#from bnetbase import *
from naive_bayes import *


test_multiply = True
test_restrict = True
test_sum = True
test_normalize = True
test_ve = True
test_nb = True


#  E,B,S,W,G example
E, B, S, G, W = Variable('E', ['e', '-e']), Variable('B', ['b', '-b']), Variable('S', ['s', '-s']), Variable('G', ['g', '-g']), Variable('W', ['w', '-w'])
FE, FB, FS, FG, FW = Factor('P(E)', [E]), Factor('P(B)', [B]), Factor('P(S|E,B)', [S, E, B]), Factor('P(G|S)', [G,S]), Factor('P(W|S)', [W,S])
multiply_result, sum_out_result = Factor('P(E) X P(B)', [E, B]), Factor('P(S|E,B) summed', [S, E])
restrict_result, normalize_result = Factor('P(S|E,B) restricted', [E, B]), Factor('P(W|S)', [W,S])

FE.add_values([['e',0.27], ['-e', 0.73]])
FB.add_values([['b', 0.41], ['-b', 0.59]])
FS.add_values([['s', 'e', 'b', .33], ['s', 'e', '-b', .08], ['s', '-e', 'b', 0],['s', '-e', '-b', .15],
                 ['-s', 'e', 'b', .67], ['-s', 'e', '-b', .92], ['-s', '-e', 'b', 1],['-s', '-e', '-b', .85]])
FG.add_values([['g', 's', 0.78], ['g', '-s', 0], ['-g', 's', 0.22], ['-g', '-s', 1]])
FW.add_values([['w', 's', 0.6], ['w', '-s', .39], ['-w', 's', 0.4], ['-w', '-s', 0.61]])
multiply_result.add_values([['e', 'b', 0.1107], ['e', '-b', 0.1593], ['-e', 'b', 0.2993], ['-e', '-b', 0.4307]])
sum_out_result.add_values([['s', 'e', 0.41], ['s', '-e', 0.15], ['-s', 'e', 1.59], ['-s', '-e', 1.85]])
restrict_result.add_values([['e', 'b', 0.67], ['e', '-b', 0.92], ['-e', 'b', 1], ['-e', '-b', 0.85]])
normalize_result.add_values([['w', 's', 0.3], ['w', '-s', 0.195], ['-w', 's', 0.2], ['-w', '-s', 0.305]])
ve_result = [0.0, 1.0]


SampleBN = BN('SampleBN', [E,B,S,G,W], [FE,FB,FS,FG,FW])
SampleBN_ = BN('SampleBN_', [E,B,S,G,W], [FE,FB,FS,FG,FW])


def nearly_equal(x, y):
    return abs(x-y) < 0.0001

def test_multiply_fun():
    print("\nMultiply Factors Test 1 ... ", end='')
    try:
      factor = multiply([FB, FE])
      tests = []
      values = []
      for e_val in E.domain():
        for b_val in B.domain():
          try:
            value = factor.get_value([e_val, b_val])
            values.append(value)
          except ValueError:
            value = factor.get_value([b_val, e_val])
            values.append(value)
          tests.append(nearly_equal(value, multiply_result.get_value([e_val, b_val])))
      if all(tests):
        print("passed.")
      else:
        print("failed.")
      print('FB = ', FB.values, 'FE = ', FE.values, 'FB * FE = ', factor.values)
    except Exception as e:
      print("error.", e)
      print('FB = ', FB.values, 'FE = ', FE.values, 'FB * FE = ', factor.values)


def test_sum_fun():
  try:
    print("\nSum Out Variable Test 1 ....", end='')
    factor = sum_out(FS, B)
    factor_ = sum_out_result
    values = (factor.get_value(["s", "e"]), factor.get_value(["s", "-e"]), factor.get_value(["-s", "e"]), factor.get_value(["-s", "-e"]))
    tests = (abs(values[0] - factor_.get_value(["s", "e"])) < 0.001, abs(values[1] - factor_.get_value(["s", "-e"])) < 0.001, abs(values[2] - factor_.get_value(["-s", "e"])) < 0.001, abs(values[3] - factor_.get_value(["-s", "-e"])) < 0.001)
    if all(tests):
      print("passed.")
    else:
      print("failed.")
    print('FS = ', FS.values, 'sum out B from FS = ', factor.values, 'real = ', factor_.values)
  except Exception as e:
    print("error.", e)
    print('FS = ', FS.values, 'sum out B from FS = ', factor.values, 'real = ', factor_.values)


def test_restrict_fun():
  try:
    print("\nRestrict Factor Test 1 ...", end='')
    factor = restrict(FS, S, '-s')
    factor_ = restrict_result
    passed = True
    if not nearly_equal(factor.get_value(['e', 'b']), factor_.get_value(['e', 'b'])):
      passed = False
    elif not nearly_equal(factor.get_value(['e', '-b']), factor_.get_value(['e', '-b'])):
      passed = False
    elif not nearly_equal(factor.get_value(['-e', 'b']), factor_.get_value(['-e', 'b'])):
      passed = False
    elif not nearly_equal(factor.get_value(['-e', '-b']), factor_.get_value(['-e', '-b'])):
      passed = False
    if passed:
      print("passed.")
    else:
      print ("failed.")
    print('FS = ', FS.values, 'restrict FS on S=-s: ', factor.values)
  except Exception as e:
    print("error.", e)
    print('FS = ', FS.values, 'restrict FS on S=-s: ', factor.values)


def test_normalize_fun():
  try:
    print("\nNormalize Factor Test ...", end='')
    factor = normalize(FW)
    factor_ = normalize_result
    passed = True
    if not nearly_equal(factor.get_value(['w', 's']), factor_.get_value(['w', 's'])):
      passed = False
    elif not nearly_equal(factor.get_value(['-w', 's']), factor_.get_value(['-w', 's'])):
      passed = False
    elif not nearly_equal(factor.get_value(['w', '-s']), factor_.get_value(['w', '-s'])):
      passed = False
    elif not nearly_equal(factor.get_value(['-w', '-s']), factor_.get_value(['-w', '-s'])):
      passed = False
    if passed:
      print("passed.")
    else:
      print ("failed.")
    print('FW = ', FW.values, 'normalized FW = ', factor.values)
  except Exception as e:
    print("error.", e)
    print('FW = ', FW.values, 'normalized FW = ', factor.values)


def test_ve_fun():
    print("\nVE Tests .... ")
    try:
      SampleBN = BN('SampleBN', [E,B,S,G,W], [FE,FB,FS,FG,FW])
      SampleBN_ = BN('SampleBN_', [E,B,S,G,W], [FE,FB,FS,FG,FW])
      S.set_evidence('-s')
      W.set_evidence('w')
      probs3_ = ve_result
      probs3 = ve(SampleBN, G, [S,W]).values
      print("Test 1 ....", end = '')
      if abs(probs3[0] - probs3_[0]) < 0.0001 and abs(probs3[1] - probs3_[1]) < 0.0001:
        print("passed.")
      else:
        print("failed.")
      print('Bayes Net: FE = ', FE.values, 'FB = ', FB.values, 'FS = ', FS.values, 'FG = ', FG.values, 'FW = ', FW.values)
      print('VE with query G and evidence S=-s, W=w:', probs3)
    except Exception as e:
      print('Bayes Net: FE = ', FE.values, 'FB = ', FB.values, 'FS = ', FS.values, 'FG = ', FG.values, 'FW = ', FW.values)
      print('VE with query G and evidence S=-s, W=w:', probs3)
      print("error.", e)


def test_explore_fun():
    print("\nExplore Test")
    try:
      nb = naive_bayes_model('data/adult-train.csv')
      one = explore(nb, 1)
      two = explore(nb, 2)
      three = explore(nb, 3)
      four = explore(nb, 4)
      five = explore(nb, 5)
      six = explore(nb, 6)
      passed = True
      if one < 0 or one > 100:
          print("failed one")
          passed = False
      if two < 0 or two > 100:
          print("failed two")
          passed = False
      if three < 0 or three > 100:
          print("failed three")
          passed = False
      if four < 0 or four > 100:
          print("failed four")
          passed = False
      if five < 0 or five > 100:
          print("failed five")
          passed = False
      if six < 0 or six > 100:
          print("failed six")
          passed = False
      if passed:
        print ("passed.")
    except Exception as e:
      print("error.", e)

if __name__ == '__main__':
    try:
        if test_multiply: test_multiply_fun()
    except:
        print("failed two multiply tests")
    try:
        if test_sum: test_sum_fun()
    except:
        print("failed two sum tests")
    try:
        if test_restrict: test_restrict_fun()
    except:
        print("failed two restrict tests")
    try:
        if test_normalize: test_normalize_fun()
    except:
        print("failed normalize test")
    try:
        if test_ve: test_ve_fun()
    except:
        print("failed 4 ve tests")
    test_explore_fun()
