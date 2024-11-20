import string
import random
from python_enigma import enigma
import json

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

def encode_strings(words : list, catalog, stecker, rotors, reflector, operator, word_length, stator):
  encoded = []
  machine = enigma.Enigma(catalog=catalog, stecker=stecker, 
                          rotors=rotors, reflector=reflector, 
                          operator=operator, word_length=word_length, 
                          stator=stator)
  for word in words:
    data = {}
    machine.set_wheels("ABC")
    data["plain"] = word
    data["encoded"] = machine.parse(word)
    data["catalog"] = catalog
    data["stecker"] = stecker
    data["rotors"] = rotors
    data["reflector"] = reflector
    data["operator"] = operator
    data["word_length"] = word_length
    data["stator"] = stator
    encoded += [data]
  return encoded

def random_data_gen(strs : int):
  words = random_text(strs)
  # default settings from helloworld.py
  use_these = [("I", "A"), ("II", "B"), ("III", "C")]
  encoded = encode_strings(words, catalog="default", stecker="AQ BJ",rotors=use_these, reflector="Reflector B", operator=True, word_length=5, stator="military")
  with open("random_data.json", mode="w") as file:
    for e in encoded:
      json.dump(e, file, indent="\t")