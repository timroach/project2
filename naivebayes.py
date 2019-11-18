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
        self.totalset = np.array(spamlist)
        self.trainingspam = np.array(isspam[:906])
        self.trainingnot = np.array(notspam[:1359])
        self.testingspam = np.array(isspam[907:907+907])
        self.testingnot = np.array(notspam[1360:1360+1359])
        # Total size (number of examples) of each set
        self.trainingsize = self.trainingspam.shape[0] + self.trainingnot.shape[0]
        self.testingsize = self.testingspam.shape[0] + self.testingnot.shape[0]
        self.features = self.trainingspam.shape[1] - 1

    # Compute prior probabilities of training set features
    def computepriors(self):
        # Result arrays
        spamfeaturemean = np.zeros(self.features)
        notfeaturemean = np.zeros(self.features)
        spamfeaturedev = np.zeros(self.features)
        notfeaturedev = np.zeros(self.features)
        totalfeaturemean = np.zeros(self.features)
        totalfeaturedev = np.zeros(self.features)
        for i in range(0, self.features):
            # Get values for each feature into an array
            spamcolumn = self.trainingspam[:, i]
            notcolumn = self.trainingnot[:, i]
            totalcolumn = self.totalset[:, i]
            # Find mean, store in result array
            spamfeaturemean[i] = np.mean(spamcolumn)
            notfeaturemean[i] = np.mean(notcolumn)
            totalfeaturemean[i] = np.mean(totalcolumn)
            # Find std dev, if 0, set to 0.0001, store it
            spamdev = np.std(spamcolumn)
            if spamdev == 0:
                spamdev = 0.0001
            spamfeaturedev[i] = spamdev
            notdev = np.std(notcolumn)
            if notdev == 0:
                notdev = 0.0001
            notfeaturedev[i] = notdev
            totaldev = np.std(totalcolumn)
            if totaldev == 0:
                totaldev = 0.0001
            totalfeaturedev[i] = totaldev
        resultdict = {"spammean": spamfeaturemean,
                      "notmean": notfeaturemean,
                      "spamdev": spamfeaturedev,
                      "notdev": notfeaturedev,
                      "totalmean": totalfeaturemean,
                      "totaldev": totalfeaturedev}
        return resultdict


def main():
    myclassifier = NaiveBayes()
    priors = myclassifier.computepriors()
    print("test")
    sys.exit(0)

if __name__ == "__main__":
    main()