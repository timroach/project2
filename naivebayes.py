import sys
import numpy as np


class NaiveBayes:

    def __init__(self):
        # Reads spam dataset from file and
        # store set into spamlist, while
        # converting each value float except
        # last value, which is int for class
        # (1 = spam, 0 = not spam)
        spamlist = []
        isspam = []
        notspam = []
        with open("spambase.data") as spamfile:
            rawdata = spamfile.read()
            listofstrings = rawdata.split('\n')
            for string in listofstrings:
                if string:
                    values = string.split(',')
                    spamornot = int(values[-1])
                    values = [float(i) for i in values[:-1]]
                    values.append(spamornot)
                    spamlist.append(values)
                    # Create sublists of spam and not-spam
                    if values[-1] is 1:
                        isspam.append(values)
                    else:
                        notspam.append(values)
        # Create training and testing sets
        # each with 906 (40%) spam, 1359 (60%) not-spam
        # 2265 total examples each set
        trainingset = isspam[:906] + notspam[:1359]
        self.trainingset = np.array(trainingset)
        testingset = isspam[907:907+907] + notspam[1360:1360+1359]
        self.testingset = np.array(testingset)


def main():
    myclassifier = NaiveBayes()
    print("test")
    sys.exit(0)

if __name__ == "__main__":
    main()