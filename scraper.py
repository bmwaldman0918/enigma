from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
import string

BEN_LAPTOP_DRIVER_PATH = "C:\\Users\\bmwal_sbkb7fk\\chromedriver-win64"

def init_driver(computer):
  match computer:
    case "BEN_LAPTOP":
      s = webdriver.ChromeService(exeutable_path=BEN_LAPTOP_DRIVER_PATH)
      driver = webdriver.Chrome(service=s)
  driver.implicitly_wait(5)
  return driver

def get_text_from_page(driver: webdriver.Chrome, url):
  driver.get(url)
  elems = driver.find_elements(by=By.TAG_NAME, value="p")
  processed = []
  for e in elems:
    text = e.get_attribute('innerText').split()
    for t in text:
      s = t.lower()
      s = [c for c in s if c in string.ascii_letters]
      s = ''.join(s)
      processed += [s]
  return processed

d = init_driver("BEN_LAPTOP")
url = "https://en.wikipedia.org/wiki/PageRank"
# print(get_text_from_page(d, url))