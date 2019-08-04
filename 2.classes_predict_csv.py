import cv2, os
import numpy as np
import tensorflow as tf
import csv

with tf.gfile.FastGFile("./model/train-model.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

image_size = 128
num_channels = 3

classes = [line.rstrip() for line
                   in tf.gfile.GFile("./model/train-model.csv")]

with tf.Session() as sess:
    
    saver = tf.train.import_meta_graph('./model/train-model.meta')
    saver.restore(sess, "./model/train-model")
    
    y_pred = sess.graph.get_tensor_by_name('y_pred:0')
    x = sess.graph.get_tensor_by_name("x:0")
      
    path = 'testing_data'
    files = sorted(os.listdir(path))
    
    for filename in files:
            
        with open('testing_data/' + filename + '.csv', 'a') as csvfile:
               writer = csv.writer(csvfile, delimiter=',')
               writer.writerow(["FILENAME","ACTUAL", "PREDICTED"])
               
        images_all = sorted(os.listdir(path+'/'+filename))
        
        for images in images_all:
            
            image = cv2.imread(path+'/'+filename+'/'+images)
            if image is not None:            

                frame = image.copy()
                image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
                image = image.astype(np.float32)
                image = np.multiply(image, 1.0 / 255.0)
                image = np.array(image)
                x_batch = image.reshape(1, image_size, image_size, num_channels)
    
                feed_dict_testing = {x: x_batch}
                result=sess.run(y_pred, feed_dict=feed_dict_testing)
        
                index = np.argmax(result) 
                               
                pred_class = classes[index]
                accuracy = result[0][index]*100
                                    
                print ("Class: ", pred_class, " | Accuracy: ", accuracy ," %")
    		
#                cv2.imshow('frame', frame)
#               
#                k = cv2.waitKey(1)
#                if k == 27:
#                    break
                
                with open('testing_data/' + filename + '.csv', 'a') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',')
                        row = filename + "," + pred_class + "\n"
                        writer.writerow([images,filename, pred_class])
            else:
                break
            
#cv2.destroyAllWindows() 
#cv2.VideoCapture().release()
