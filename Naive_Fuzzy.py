from audioop import avg
import numpy as np
from PIL import Image
import statistics

path = "./Example2/image.jpg"

def naive_bayes_classifier(input_filepath):
  # input_filepath is the full file path to a JPG file containing an RGB image
  # Outputs a string (classified class name) and a list of probabilities of all classes


  img = np.asarray(Image.open(path))


  classes = ["tundra", "forest", "desert", "ocean"]




  # Given data
  𝑡𝑢𝑛𝑑𝑟𝑎 = 0.03
  𝑓𝑜𝑟𝑒𝑠𝑡 = 0.10
  𝑑𝑒𝑠𝑒𝑟𝑡 = 0.11
  𝑜𝑐𝑒𝑎𝑛 = 0.76



  redGivenT𝑢𝑛𝑑𝑟𝑎 = 0.85
  redGivenF𝑜𝑟𝑒𝑠𝑡 = 0.53
  redGivenD𝑒𝑠𝑒𝑟𝑡 = 0.94
  redGivenO𝑐𝑒𝑎𝑛 = 0.18
  greenGivenT𝑢𝑛𝑑𝑟𝑎 = 0.71
  greenGivenF𝑜𝑟𝑒𝑠𝑡 = 0.88
  greenGivenD𝑒𝑠𝑒𝑟𝑡 = 0.06
  greenGivenO𝑐𝑒𝑎𝑛 = 0.27
  blueGivenT𝑢𝑛𝑑𝑟𝑎 = 0.89
  blueGivenF𝑜𝑟𝑒𝑠𝑡 = 0.12
  blueGivenD𝑒𝑠𝑒𝑟𝑡 = 0.03
  blueGivenO𝑐𝑒𝑎𝑛 = 0.98

  notRedGivenTundra = 1- redGivenTundra
  notRedGivenForest = 1 - redGivenForest
  notRedGivenDesert = 1-redGivenDesert
  notRedGivenOcean = 1 - redGivenOcean

  notGreenGivenTundra = 1 - greenGivenTundra
  notGreenGivenForest = 1-greenGivenForest
  notGreenGivenDesert = 1-greenGivenDesert
  notGreenGivenOcean = 1-greenGivenOcean

  notBlueGivenTundra = 1-blueGivenTundra
  notBlueGivenForest = 1-blueGivenForest
  notBlueGivenDesert = 1-blueGivenDesert
  notBlueGivenOcean = 1-blueGivenOcean
  






  # Redness of image
  redness =img[:,:,0].mean()

  # Greeness of image
  greeness = img[:,:,1].mean()

  # Blueness of image
  blueness = img[:,:,2].mean()


  # Coding every possible condition seprately
  if redness>128 and greeness>128 and blueness>128:

    ProbEvidence = (redGivenTundra * greenGivenTundra * blueGivenTundra * tundra) + (redGivenForest * greenGivenForest * blueGivenForest * forest) + (redGivenDesert * greenGivenDesert * blueGivenDesert * desert) + (redGivenOcean * greenGivenOcean * blueGivenOcean * ocean)

    ProbTundra = (redGivenTundra * greenGivenTundra * blueGivenTundra * tundra) / ProbEvidence
    ProbForest = (redGivenForest * greenGivenForest * blueGivenForest * forest) / ProbEvidence
    ProbDesert = (redGivenDesert * greenGivenDesert * blueGivenDesert * desert) / ProbEvidence
    ProbOcean = (redGivenOcean * greenGivenOcean * blueGivenOcean * ocean) / ProbEvidence


  elif redness>128 and greeness>128 and blueness<=128:
    
    ProbEvidence = (redGivenTundra * greenGivenTundra * notBlueGivenTundra * tundra) + (redGivenForest * greenGivenForest * notBlueGivenForest * forest) + (redGivenDesert * greenGivenDesert * notBlueGivenDesert * desert) + (redGivenOcean * greenGivenOcean * notBlueGivenOcean * ocean)

    ProbTundra = (redGivenTundra * greenGivenTundra * (notBlueGivenTundra) * tundra) / ProbEvidence
    ProbForest = (redGivenForest * greenGivenForest * (notBlueGivenForest) * forest) / ProbEvidence
    ProbDesert = (redGivenDesert * greenGivenDesert * (notBlueGivenDesert) * desert) / ProbEvidence
    ProbOcean = (redGivenOcean * greenGivenOcean * (notBlueGivenOcean) * ocean) / ProbEvidence

  elif redness>128 and greeness<=128 and blueness>128:

    ProbEvidence = (redGivenTundra * notGreenGivenTundra * blueGivenTundra * tundra) + (redGivenForest * notGreenGivenForest * blueGivenForest * forest) + (redGivenDesert * notGreenGivenDesert * blueGivenDesert * desert) + (redGivenOcean * notGreenGivenOcean * blueGivenOcean * ocean)

    ProbTundra = (redGivenTundra * notGreenGivenTundra * blueGivenTundra * tundra) / ProbEvidence
    ProbForest = (redGivenForest * notGreenGivenForest * blueGivenForest * forest) / ProbEvidence
    ProbDesert = (redGivenDesert * notGreenGivenDesert * blueGivenDesert * desert) / ProbEvidence
    ProbOcean = (redGivenOcean * notGreenGivenOcean * blueGivenOcean * ocean) / ProbEvidence

  elif redness<=128 and greeness>128 and blueness>128:

    ProbEvidence = (notRedGivenTundra * greenGivenTundra * blueGivenTundra * tundra) + (notRedGivenForest * greenGivenForest * blueGivenForest * forest) + (notRedGivenDesert * greenGivenDesert * blueGivenDesert * desert) + (notRedGivenOcean * greenGivenOcean * blueGivenOcean * ocean)

    ProbTundra = ((notRedGivenTundra) * greenGivenTundra * blueGivenTundra * tundra) / ProbEvidence
    ProbForest = ((notRedGivenForest) * greenGivenForest * blueGivenForest * forest) / ProbEvidence
    ProbDesert = ((notRedGivenDesert) * greenGivenDesert * blueGivenDesert * desert) / ProbEvidence
    ProbOcean = ((notRedGivenOcean) * greenGivenOcean * blueGivenOcean * ocean) / ProbEvidence

  elif redness>128 and greeness<=128 and blueness<=128:

    ProbEvidence = (redGivenTundra * notGreenGivenTundra * notBlueGivenTundra * tundra) + (redGivenForest * notGreenGivenForest * notBlueGivenForest * forest) + (redGivenDesert * notGreenGivenDesert * notBlueGivenDesert * desert) + (redGivenOcean * notGreenGivenOcean * notBlueGivenOcean * ocean)

    ProbTundra = (redGivenTundra * (notGreenGivenTundra) * (notBlueGivenTundra) * tundra) / ProbEvidence
    ProbForest = (redGivenForest * (notGreenGivenForest) * (notBlueGivenForest) * forest) / ProbEvidence
    ProbDesert = (redGivenDesert * (notGreenGivenDesert) * (notBlueGivenDesert) * desert) / ProbEvidence
    ProbOcean = (redGivenOcean * (notGreenGivenOcean) * (notBlueGivenOcean) * ocean) / ProbEvidence



  elif redness<=128 and greeness<=128 and blueness>128:

    ProbEvidence = (notRedGivenTundra * notGreenGivenTundra * blueGivenTundra * tundra) + (notRedGivenForest * notGreenGivenForest * blueGivenForest * forest) + (notRedGivenDesert * notGreenGivenDesert * blueGivenDesert * desert) + (notRedGivenOcean * notGreenGivenOcean * blueGivenOcean * ocean)

    ProbTundra = ((notRedGivenTundra) * (notGreenGivenTundra) * blueGivenTundra * tundra) / ProbEvidence
    ProbForest = ((notRedGivenForest) * (notGreenGivenForest) * blueGivenForest * forest) / ProbEvidence
    ProbDesert = ((notRedGivenDesert) * (notGreenGivenDesert) * blueGivenDesert * desert) / ProbEvidence
    ProbOcean = ((notRedGivenOcean )* (notGreenGivenOcean) * blueGivenOcean * ocean) / ProbEvidence

  elif redness<=128 and greeness>128 and blueness<=128:

    ProbEvidence = (notRedGivenTundra * greenGivenTundra * notBlueGivenTundra * tundra) + (notRedGivenForest * greenGivenForest * notBlueGivenForest * forest) + (notRedGivenDesert * greenGivenDesert * notBlueGivenDesert * desert) + (notRedGivenOcean * greenGivenOcean * notBlueGivenOcean * ocean)

    ProbTundra = ((notRedGivenTundra) * greenGivenTundra * (notBlueGivenTundra) * tundra) / ProbEvidence
    ProbForest = ((notRedGivenForest) * greenGivenForest * (notBlueGivenForest) * forest) / ProbEvidence
    ProbDesert = ((notRedGivenDesert) * greenGivenDesert * (notBlueGivenDesert) * desert) / ProbEvidence
    ProbOcean = ((notRedGivenOcean) * greenGivenOcean * (notBlueGivenOcean) * ocean) / ProbEvidence

  else:

    ProbEvidence = (notRedGivenTundra * notGreenGivenTundra * notBlueGivenTundra * tundra) + (notRedGivenForest * notGreenGivenForest * notBlueGivenForest * forest) + (notRedGivenDesert * notGreenGivenDesert * notBlueGivenDesert * desert) + (notRedGivenOcean * notGreenGivenOcean * notBlueGivenOcean * ocean)

    ProbTundra = ((notRedGivenTundra) * (notGreenGivenTundra) * (notBlueGivenTundra) * tundra) / ProbEvidence
    ProbForest = ((notRedGivenForest) * (notGreenGivenForest) * (notBlueGivenForest) * forest) / ProbEvidence
    ProbDesert = ((notRedGivenDesert) * (notGreenGivenDesert) * (notBlueGivenDesert) * desert) / ProbEvidence
    ProbOcean = ((notRedGivenOcean) * (notGreenGivenOcean) * (notBlueGivenOcean) * ocean) / ProbEvidence

  
  class_probabilities = [ProbTundra, ProbForest, ProbDesert, ProbOcean]

  most_likely_class = classes[class_probabilities.index(max(class_probabilities))]



  # most_likely_class is a string indicating the most likely class, either “tundra”, “forest”, “desert”, or “ocean”
  # class_probabilities is a five element list indicating the probability of each class in the order [tundra probability, forest probability, desert probability, ocean probability]
  return most_likely_class, class_probabilities


def fuzzy_classifier(input_filepath):



  def fuzzify(a, b, c, d, x):
    if x<=a:
      memValue = 0

    elif x > a and x < b:
      memValue = (x - a) / (b - a)

    elif x >= b and x <= c:
      memValue = 1
    
    elif x > c and x < d:
      memValue = (d-x) / (d-c)
    
    elif x>=d and x <= 255:
      memValue = 0  

    return memValue


  # input_filepath is the full file path to a JPG file containing an RGB image
  # Hint: here is how you can read an image as a numpy array
  
  img = np.asarray(Image.open(input_filepath))


  # Redness of image
  redness = img[:,:,0].mean()

  # Greeness of image
  greeness = img[:,:,1].mean()

  # Blueness of image
  blueness = img[:,:,2].mean()


  # 1st entry if for low, 2nd for mediam, 3rd for High
  redFuzzyList = [fuzzify(0,0,85,125, redness), fuzzify(85,125,130,190, redness), fuzzify(130, 190, 255, 255, redness)]


  # 1st entry if for low, 2nd for mediam, 3rd for High
  greenFuzzyList = [fuzzify(0,0,60,120, greeness), fuzzify(60, 120, 125,185, greeness), fuzzify(125, 185, 255, 255, greeness)]



  # 1st entry if for low, 2nd for mediam, 3rd for High
  blueFuzzyList = [fuzzify(0,0,55,130, blueness), fuzzify(55,130,140,190, blueness), fuzzify(140, 190, 255, 255, blueness)]



  # Calculating membershipValue for every class
  tundraMembershipValue = min(redFuzzyList[2], greenFuzzyList[2], blueFuzzyList[2])

  forestMembershipValue = min(max(redFuzzyList[0], redFuzzyList[1]), greenFuzzyList[2], max(blueFuzzyList[0], blueFuzzyList[1]))

  desertMembershipValue = min(redFuzzyList[2], greenFuzzyList[0], blueFuzzyList[0])

  oceanMembershipValue = min(redFuzzyList[0], blueFuzzyList[2])

  classes = ["tundra", "forest", "desert", "ocean"]

  class_memberships = [tundraMembershipValue, forestMembershipValue, desertMembershipValue, oceanMembershipValue]

  highest_membership_class = classes[class_memberships.index(max(class_memberships))]


  



  # highest_membership_class is a string indicating the highest membership class, either “tundra”, “forest”, “desert”, or “ocean”
  # class_memberships is a four element list indicating the membership in each class in the order [tundra value, forest value, desert value, ocean value]
  return highest_membership_class, class_memberships





# prediction, probs = naive_bayes_classifier(path)

# print(prediction, probs)

prediction, probs = fuzzy_classifier(path)

print(prediction, probs)


prediction, probs = naive_bayes_classifier(path)
print(prediction, probs)
