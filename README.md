# keras_image_classifier
Classify images, based on training data

Usage: 
1. create folder with:
    - folder with training data (one folder for each type)
    - folder with images to be classified
    - this script
2. set required parameters:
    - data_dir = (relative) folder with traing/validation images ('document_images')
    - epoch = number of passes of the entire training dataset in the machine learning algorithm ('10')
    - path = (relative) folder with images that need to be predicted ('test')
3. in terminal: '$ python document_classifier_keras.py'
4. results are written to csv file 'predicted_image_types.csv'

see https://www.tensorflow.org/tutorials/images/classification
