

from wordfreq import get_frequency_dict
import numpy as np


#five_dict = {"HELLO": 0.1, "FLUFF": 0.5, "FRETS": 0.4}


def normalize(distribution):
  psum = 0
  for k in distribution:
    psum += distribution[k]
  if psum == 0:
    return None
  return {k: distribution[k]/psum for k in distribution}


all_words = get_frequency_dict('en', wordlist='small')
five_dict = {k: all_words[k] for k in all_words if len(k) == 5}
five_dict = normalize({k: five_dict[k] for k in five_dict if five_dict[k] > 1e-5})
print(len(five_dict))



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
def matches_coloring(candidate, guess, coloring):
  return get_coloring_from_guess(candidate, guess) == coloring

# returns a normalized conditional distribution given a guess and coloring
def prune_candidates(candidates, guess, coloring):
  psum = 0
  new_candidates = {}
  for word in candidates:
    if matches_coloring(word, guess, coloring):
      p = candidates[word]
      new_candidates[word] = p
  # normalized
  return normalize(new_candidates)

def compute_entropy(distribution):
  s = 0
  for k in distribution:
    p = distribution[k]
    s += (- np.log(p) * p)
  return s

# candidates: a dict of probabilities. should sum to 1.
def get_best_guess(candidates):

  current_entropy = compute_entropy(candidates)
  expected_entropy_reduction = {k: 0 for k in candidates}

  # sum over all words in candidates according to their weights
  for true_word in candidates:
    for guess in candidates:
      # what the distribution would be after this guess, for this true word
      pruned = prune_candidates(candidates, guess, get_coloring_from_guess(true_word, guess))
      entropy_reduction = current_entropy - compute_entropy(pruned)
      # in computing expected value, need to weight this by the true word probability
      p_true_word = candidates[true_word]
      expected_entropy_reduction[guess] += p_true_word * entropy_reduction

  best = 0
  best_key = None
  for k in candidates:
    print(k, expected_entropy_reduction[k])
    if expected_entropy_reduction[k] > best:
      best_key = k
      best = expected_entropy_reduction[k]

  # debugging
  print("Best entropy reduction = ", best)
  return best_key

def coloring_from_string(coloring_string):
  if len(coloring_string) != 5:
    return None
  result = []
  for c in coloring_string:
    if c < '0' or c > '9':
      return None
    result.append(int(c))
  return result



if __name__ == "__main__":
  print("Welcome! This Wordle utility will suggest a guess for you.")
  print("After submitting, please type the coloring of that guess and hit enter.")
  print("Write 0 for black, 1 for gold, and 2 for green. So, you may type 01002[enter], e.g..")
  print("If that coloring was not 22222 (all green), the utility will provide you the next guess, and so on.")

  candidates = five_dict.copy()
  while True:
    print("Total candidates = ", len(candidates))
    guess = get_best_guess(candidates)
    print("The utility suggests you guess: " + guess)
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
    candidates = prune_candidates(candidates, guess, coloring)
    if not candidates:
      print("Oh no! Wordle has a word in mind that is not in our dictionary.")
      break

