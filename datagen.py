import string
import random
from python_enigma import enigma
import json
import scraper

MIN_STR_LEN = 1
MAX_STR_LEN = 25

# generate random text
def random_text(string_quantity : int):
  strs = []
  for _ in range(string_quantity):
    length = random.choice(range(MIN_STR_LEN, MAX_STR_LEN))
    s = "".join(random.choices(string.ascii_lowercase, k=length))
    strs += [scraper.pad(str(s))]
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
    file.write(json.dumps(encoded, indent="\t"))

def scraped_data_gen(urls : dict, driver):
  all_text = []
  for u in urls:
    url = urls[u]
    text = scraper.get_text_from_page(driver, url)
    all_text += text
    use_these = [("I", "A"), ("II", "B"), ("III", "C")]
  encoded = encode_strings(all_text, catalog="default", stecker="AQ BJ",rotors=use_these, reflector="Reflector B", operator=True, word_length=5, stator="military")
  with open("scraped_data.json", mode="w") as file:
    file.write(json.dumps(encoded, indent="\t"))

d = scraper.init_driver("BEN_LAPTOP")
urls = {}
urls["befunge"] = "https://en.wikipedia.org/wiki/Befunge"
urls["brainfuck"] = "https://en.wikipedia.org/wiki/Brainfuck"
urls["fractran"] = "https://en.wikipedia.org/wiki/FRACTRAN"
urls["intercal"] = "https://en.wikipedia.org/wiki/INTERCAL"
urls["malbolge"] = "https://en.wikipedia.org/wiki/Malbolge"
urls["obama"] = "https://en.wikipedia.org/wiki/Barack_Obama"
urls["churchill"] = "https://en.wikipedia.org/wiki/Winston_Churchill"
urls["ovechkin"] = "https://en.wikipedia.org/wiki/Alexander_Ovechkin"
scraped_data_gen(urls, d)
random_data_gen(int(10e4))