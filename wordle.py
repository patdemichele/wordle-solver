from wordfreq import get_frequency_dict
import numpy as np
import sys

WORDLE_GUESSES_FILENAME = "wordle_allowed_guesses.txt"
WORDLE_ANSWERS_FILENAME = "wordlist_solutions.txt"

WORDLE_TEST_FILENAME = "wordle_archive.txt"

# for the unchanging uniform and english distributions, we can save time by precomputing the first guess.
# UNIFORM Here are your recommended guesses, best guess first:  ['alter', 'trace', 'raise', 'irate', 'hater']
# ENGLISH Here are your recommended guesses, best guess first:  ['slate', 'saner', 'stone', 'least', 'roast']
precomputed_first_guesses = {"english": "slate", "uniform": "alter"}


# distribution size such that beyond this size we "compactify" the distribution by sampling it, in certain cases.
MAX_DISTRIBUTION_SIZE = 200

def normalize(distribution):
  if len(distribution) == 0:
    return {}
  psum = 0
  for k in distribution:
    psum += distribution[k]
  if psum == 0:
    return None
  return {k: distribution[k]/psum for k in distribution}


def compact_distribution(distribution):
  # if the distribution is small, just use the distribution
  if len(distribution) <= MAX_DISTRIBUTION_SIZE:
    return distribution
  # otherwise, sample it
  a = [k for k in distribution]
  p = [distribution[k] for k in a]
  choices = np.random.choice(a, MAX_DISTRIBUTION_SIZE, replace=True, p=p)
  result = {}
  for c in choices:
    result[c] = result.get(c, 0) + 1.0/MAX_DISTRIBUTION_SIZE
  return result

# used in get_coloring_from_guess
def get_hist(word):
  h = {}
  for c in word:
    h[c] = h.get(c,0) + 1
  return h

# the way the colorings work is actually a bit subtle.
# first, mark all the green letters (that part is obvious).
# after that, a letter appears gold in the guess word if
# there is at least one more letter in the true_word of that letter.
def get_coloring_from_guess(true_word, guess):
  assert(len(true_word) == 5 and len(guess) == 5)
  result = [0, 0, 0, 0, 0]
  true_hist = get_hist(true_word)

  # mark the greens  
  for i in range(5):
    if guess[i] == true_word[i]:
      result[i] = 2
      c = guess[i]
      true_hist[c] -= 1

  # mark unmatched letters in the guess as yellow if more letters of that type remain in the true word
  for i in range(5):
    if result[i] != 2: # not a match
      c = guess[i]
      if true_hist.get(c,0) > 0:
        result[i] = 1
        true_hist[c] -= 1
  
  return result
  

# Boolean: whether candidate could be the true word
# if guess yields coloring
def matches_coloring(true_word, guess, coloring):
  return get_coloring_from_guess(true_word, guess) == coloring

# computes the entropy of a distribution
def compute_entropy(distribution):
  s = 0
  for k in distribution:
    p = distribution[k]
    s += (- np.log(p) * p)
  return s

# prunes candidates. will use compactified distribution unless approximate=False
def prune_candidates(candidates, guess, coloring, approximate=True):
  marginal_dist = {}
  if approximate:
    data = compact_distribution(candidates)
  else:
    data = candidates
  for word in data:
    p_word = data[word]
    if matches_coloring(word, guess, coloring):
      marginal_dist[word] = marginal_dist.get(word, 0) + p_word
  return normalize(marginal_dist)


# computation of new entropy .
# note that prune_candidates is approximate (based on sampling) when the data set has larger than 100 words
def compute_new_entropy(candidates, true_word, guess):
  coloring = get_coloring_from_guess(true_word, guess)
  pruned = prune_candidates(candidates, guess, coloring)
  ret = compute_entropy(pruned)
  #print("new entropy = ", ret)
  return ret

# candidates: a dict of probabilities. should sum to 1.
def get_best_guesses(answer_distribution, allowed_guesses, exhaustive=False):

  current_entropy = compute_entropy(answer_distribution)

  # sum over all words in candidates according to their weights
  # use sampling
  data = compact_distribution(answer_distribution)
  # in the exhaustive case, check the quality of all guesses
  if exhaustive:
    guesses_to_check = allowed_guesses 
  # normal case: compute entropy reduction of guesses in data that are allowed guesses
  else:
    guesses_to_check = [guess for guess in data if guess in allowed_guesses]

  expected_entropy_reduction = {k:0 for k in guesses_to_check}
  
  for true_word in data:
    p_true_word = data[true_word]
    for guess in guesses_to_check:
      entropy_reduction = current_entropy - compute_new_entropy(answer_distribution, true_word, guess)
      # in computing expected value, need to weight this by the true word probability
      # use 1/N because we are sampling
      expected_entropy_reduction[guess] += p_true_word * entropy_reduction
 
  
  # invert the expected_entropy_reduction map
  reduction_amount_to_words = {}
  for word in expected_entropy_reduction:
    # no point considering words that are not allowed guesses
    if word not in allowed_guesses:
      continue
    amt = expected_entropy_reduction[word]
    # add word to list of words that have this entropy reduction
    reduction_amount_to_words[amt] = reduction_amount_to_words.get(amt, []) + [word]

  best_words = []  
  while len(best_words) < min(5, len(expected_entropy_reduction)):
    amt = max(reduction_amount_to_words)
    # list of words, with the most likely one first
    words = sorted(reduction_amount_to_words[amt], key=lambda k: answer_distribution.get(k, 0.0), reverse=True)
    best_words += words
    del reduction_amount_to_words[amt]
  best_words = best_words[:5]

  return best_words 

def coloring_from_string(coloring_string):
  if len(coloring_string) != 5:
    return None
  result = []
  for c in coloring_string:
    if c < '0' or c > '9':
      return None
    result.append(int(c))
  return result


# get the distribution of length-5 words we want to use
def get_wordfreq_distribution():
  all_words = get_frequency_dict('en', wordlist='large')
  five_dict = {k: all_words[k] for k in all_words if len(k) == 5}
  return normalize(five_dict)

def get_custom_distribution(path):
  d = {}
  # file can either be line separated words (then they are all weighted equally), or word weight\n word weight\n, etc
  done_first_line = False
  weights = False
  with open(path, 'r') as f:
    for line in f:
      sp = line.split(" ")
      # skip empty lines
      if len(sp) == 0:
        continue
      elif len(sp) == 2:
        if done_first_line and not weights:
          return None
        word = sp[0].strip().lower()
        if len(word) != 5:
          return None
        weight = float(sp[1])
        d[word] = weight
        weights = True
      elif len(sp) == 1:
        if done_first_line and weights:
          return None
        word = sp[0].strip().lower()
        d[word] = 1.0 # will be normalized
        weights = False
      else:
        return None # bad line
      done_first_line = True
  return normalize(d)

def get_wordle_uniform_distribution():
  return get_custom_distribution(WORDLE_ANSWERS_FILENAME)

def load_guesses_from_file(guesses_filename):
  allowed_guesses = set()
  f = open(guesses_filename, 'r')
  for line in f:
    guess = line.strip().lower()
    allowed_guesses.add(guess)
  f.close()
  return allowed_guesses


def report_usage_and_exit():
  print("Usage: The prior distribution is specified in first 2 or 3 args.\nValid options are: `python3 wordle.py english`, `python3 wordle.py uniform`, or `python3 wordle.py custom path/to/prior_distribution.txt`")
  print("After specifying the prior distribution, you may optionally add the flags --compute-first-guess and/or --exhaustive, which change the wordle solver's algorithm.")
  print("By default, the wordle utility will help you through a real game of wordle. Adding the flag --test will instead simulate wordle games on each word in WORDLE_TEST_FILENAME, and report statistics.")
  print("Adding the flag --testword=[five-letter-word] will simulate a single wordle game on a specific word. Do not use quotes.")
  print("Flags can be in any order after the arguments which specify the prior distribution.")
  print("The utility only pays attention to the first --testword, and ignores any --testword if --test is also specified.")
  print("See Github for more detailed meaning of each flag.")
  exit(0)

def detect_test_flag(argv):
  if "--test" in argv:
    return (True, None)
  prefix = "--testword="
  for arg in argv:
    if arg[:len(prefix)] == prefix:
      return (True, arg[len(prefix):])
  return (False, None)


def simulate_wordle_on_word(word_data, config, word):
  answer_distribution, allowed_guesses = word_data
  distribution_option, exhaustive, compute_first_guess = config
  if word not in answer_distribution or word not in allowed_guesses:
    return -1 # can't simulate this wordle
  round_number = 1
  while True:
    if round_number == 1 and (not compute_first_guess) and distribution_option in precomputed_first_guesses:
      guess = precomputed_first_guesses[distribution_option]
    else:
      guesses = get_best_guesses(answer_distribution, allowed_guesses, exhaustive)
      guess = guesses[0] # in simulation, always use best guess
    coloring = get_coloring_from_guess(word, guess)
    if coloring == [2, 2, 2, 2, 2]:
      break
    answer_distribution = prune_candidates(answer_distribution, guess, coloring, approximate=False)
    round_number += 1
  return round_number

def display_turns(turns):
  return "INVALID_WORD" if turns == -1 else turns

def test_and_report_on_specific_word(word_data, config, specific_word):
  turns = simulate_wordle_on_word(word_data, config, specific_word)
  print(specific_word, display_turns(turns))

def test_and_report_on_test_set(word_data, config):
  f = open(WORDLE_TEST_FILENAME, 'r')
  stats = []
  failures = []
  for line in f:
    word = line.strip()
    turns = simulate_wordle_on_word(word_data, config, word)
    print(word, display_turns(turns))
  f.close()
  
  print("Failures: ", failures)
  print("Means # turns on successful words: ", stats.sum()/max(1, len(stats)))
  

def play_guessing_game(word_data, config):
  print("Welcome! This Wordle utility will suggest a guess for you on each round.")
  print("After submitting to Wordle, please type the coloring of that guess and hit enter.")
  print("Write 0 for black, 1 for gold, and 2 for green. So, you may type 01002[enter], e.g..")
  print("If that coloring was not 22222 (all green), the utility will update the posterior distribution and provide you the next guess, and so on.")
  print("......................")
  print(" ")
  
  # unpack
  answer_distribution, allowed_guesses = word_data
  distribution_option, exhaustive, compute_first_guess = config

  round_number = 1
  while True:
    print("----- ROUND", round_number, "-----")

    # because round 1 is always the same guess for a given distribution, we help ourselves out by remembering the guesses.
    if round_number == 1 and (not compute_first_guess) and distribution_option in precomputed_first_guesses:
      guess = precomputed_first_guesses[distribution_option]
      print("It is Round 1, which means that you have zero information from the Wordle board.")
      print("Assuming Wordle uses words at a frequency similar to spoken English, you should guess: ", guess)
    else:
        print("Total candidate words = ", len(answer_distribution))
        print("Thinking...")
        guesses = get_best_guesses(answer_distribution, allowed_guesses, exhaustive)
        print("Here are your recommended guesses, best guess first: ", guesses)
        print("Please type the guess you ended up actually using (lowercase, no quotes). ")
        guess = input("")
    coloring = None
    while True:
      coloring = coloring_from_string(input("Ok... what coloring did this yield?"))
      if not coloring:
        print("Invalid coloring. Remember, you should type 5 numbers with no spaces, like 12121, and hit enter.")
      else:
        break
    if coloring == [2, 2, 2, 2, 2]:
      print("Nice!")
      break
    # prune candidates. approximate=False because we want to include every word
    answer_distribution = prune_candidates(answer_distribution, guess, coloring, approximate=False)
    if len(answer_distribution) == 1:
      print("Only one candidate left: you should guess ", [x for x in answer_distribution][0])
      break
    if not answer_distribution:
      print("Oh no! Wordle has a word in mind that is not supported in the prior distribution.")
      break
    print("...........")
    print(" ")
    round_number += 1


if __name__ == "__main__":

  print("Parsing command line arguments and loading distribution...")

  # parse command line arguments
  if len(sys.argv) < 2:
    report_usage_and_exit()
  distribution_option = sys.argv[1]
  if distribution_option == "english":
    answer_distribution = get_wordfreq_distribution()
  elif distribution_option == "uniform":
    answer_distribution = get_wordle_uniform_distribution()
  elif distribution_option == "custom":
    if len(sys.argv) < 3:
      report_usage_and_exit()
    answer_distribution = get_custom_distribution(sys.argv[2])
    if not answer_distribution:
      print("Error loading custom distribution. See guidelines on Github.")
      exit(0)
  else:
    report_usage_and_exit()

  allowed_guesses = load_guesses_from_file(WORDLE_GUESSES_FILENAME)

  exhaustive = "--exhaustive" in sys.argv
  compute_first_guess = "--compute-first-guess" in sys.argv
  
  word_data = (answer_distribution, allowed_guesses)
  config = (distribution_option, exhaustive, compute_first_guess)

  # decide what action to take
  do_test, specific_word = detect_test_flag(sys.argv)
  # report test on specific word
  if do_test and specific_word:
    test_and_report_on_specific_word(word_data, config, specific_word)
  # report test on all test set
  elif do_test:
    test_and_report_on_test_set(word_data, config)
  # else: wordle assist utility
  else:
    play_guessing_game(word_data, config)  

  
