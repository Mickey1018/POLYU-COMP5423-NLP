 A readme file describes the structure of your program and how to run it.
 
Structure of program:
 - **/data**
   - **train.txt**                        ---------a text file contains training data set w/ label
   - **val.txt**                          ---------a text file contains validation data set w/ label
   - **test_data.txt**                    ---------a text file contains test data set w/o label
   - **test_prediction.txt**              ---------a text file contains predicted results on test data
 - **/templates**
   - **predict.html**                     ---------a html file to prompt and retreive input sentence from user, running at the endpoint localhost:5000/     
   - **result.html**                      ---------a html file to return the predicted result with corresponding emoji, running at endpoint localhost:5000/result 
 - **application.py**                     ---------a python file for web application, launched by typing **'python application.py'** in the terminal
 - **read.py**                            ---------a python file contians function to read and retrieve training data, validation data, and test data
 - **write.py**                           ---------a python file contains function to write the predicted results on test data
 - **text_processor.py**                  ---------a python file contains function to process the text data
 - **vectorizer.py**                      ---------a python file to train vectorizer, and write the trained vectorizer into disk 
 - **feature_extraction.py**              ---------a python file to extract features from processed text
 - **classification_model.py**            ---------a python file to build machine learning model, and contains function for evaluation and prediction on future data 
 - **trained_vectorizer.sav**             ---------a sav file to store trained vectorizer
 - **trained_model.sav**                  ---------a sav file to store trained model
 - **emotion_classification.py**          ---------a **main** python file to run text data in pipline:
                                                         1. read all text data
                                                         2. text processing
                                                         3. feature extraction
                                                         4. machine learning model training
                                                         5. evaluation on validation data set
                                                         6. prediction on test data set
                                                         7. write predicted results)

