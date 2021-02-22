import os


def write_data(results):
    # define directory to store file
    filePath = 'data/test_prediction.txt'

    # delete file if exist
    if os.path.exists(filePath):
        os.remove(filePath)

    # write predicted test result into text file
    file = open('data/test_prediction.txt', 'w+')
    for result in results:
        file.write(result+'\n')
    file.close()
