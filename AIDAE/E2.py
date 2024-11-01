"""#task1 a)b)
def checkPrime(number):
    if number <= 1:
        return False
    for i in range(2,number):
        if number % i == 0:
            return False
    return True
try:
    number = input("give me a number")
    if checkPrime(int(number)):
        print("The number is a prime number")
    else:
        print("The number is not a prime number")

except:
    print("you didn't enter a number")
#print(checkPrime(9))
def histogramofFile(filename):
    file1 = open(filename, 'r')
    lines = file1.readlines()
    dict = {}
    for line in lines:
        if line in dict.keys():
            dict[line] += 1
            else:
            dict[line] = 1
    return dict
histogramofFile(words)

file1 = open()

class  SimpleLinearModel:
    def __init__(self, slop = 0, offset = 0):
        self.slop =slop
        self.offset = offset
model =
"""