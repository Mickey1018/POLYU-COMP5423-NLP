 A readme file describes the structure of your program and how to run it.
 
Structure of program:
 - **/data**
   - **train.txt**                         ---------a text file contains training data w/ label
   - **val.txt**                           ---------a text file contains validation data w/ label
   - **test_data.txt**                     ---------a text file contains test data w/o label
   - **test_prediction.txt**               ---------a text file contains predicted results on test data
 - **/templates**
   - **predict.html**                      ---------a html file to prompt and retreive input sentence from user, running at the endpoint localhost:5000/     
   - **result.html**                       ---------a html file to return the predicted result with corresponding emoji, running at endpoint localhost:5000/result 
 - **application.py**                      ---------a python file for web application, launched by typing 'python application.py' in the terminal
 - **read.py**                             ---------a python file contians function to read and retrieve training data, validation data, and test data
 - **write.py**                            ---------a python file contains function to write the predicted results on test data
 - **text_processor.py**                   ---------a 
 - **vectorizer.py**                       ---------a
 - **feature_extraction.py**               ---------a
 - **classification_model.py**             ---------a
 - **emotion_classification.py**           ---------a
 - **trained_vectorizer.sav**              ---------a
 - **trained_model.sav**                   ---------a

