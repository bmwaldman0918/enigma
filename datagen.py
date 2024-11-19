import string
import random

MIN_STR_LEN = 1
MAX_STR_LEN = 25

# generate random text
def random_text(string_quantity : int):
  strs = []
  for _ in range(string_quantity):
    length = random.choice(range(MIN_STR_LEN, MAX_STR_LEN))
    s = "".join(random.choices(string.ascii_lowercase, k=length))
    strs += [str(s)]
  return strs
# encode in enigma 
# output to json w settings as features