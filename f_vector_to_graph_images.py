
from keras.preprocessing import image 
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


 



# each channel of image for one feature vector
def G_to_img2(feature_vector, feature_vector2, counter, Node_size, edge_width, directory =   "E:"):
    
    
    """
    forms HVG, draws the graph and saves the 2D graph image
    
    Parameters
    -----------
    feature_vector:first set of features
    feature_vector2: second set of featires
    counter: the number of epoch we want to draw the HVG for
    
    Node_size: (the size of each node)
    edge_width: (the width of edges)
    directory:  (to save graph images)
    
    
    
    """

    G = hvisibility_graph1(feature_vector[1:28, counter])
    G1 = hvisibility_graph1(feature_vector2[:,counter])

    
    nx.draw(G, nx.get_node_attributes(G, 'pos'),
            node_color = list(G), vmin = 0.25, vmax = 1.75,
            width = 8 , node_size = 1000, cmap=plt.cm.gist_ncar)
    del G
    plt.savefig(directory + '\\PSD\\'+str('eeg')+str(counter)+'.jpg')
    plt.clf() #clears figure
    plt.close()
    
    nx.draw(G1, nx.get_node_attributes(G1, 'pos'),
            node_color = list(G1), vmin = 0.25, vmax = 1.75,
            width = 8 , node_size = 1300, cmap=plt.cm.jet)
    del G1
    plt.savefig(directory + '\\nonlinear\\'+ str('eog_eeg') + str(counter)+'.jpg')
    plt.clf() #clears figure
    plt.close()
    


z = 0
while (z<len(hypn)):
    G_to_img2(feature_vector = fv11 ,feature_vector2 = fv22, counter = z, Node_size = 600, edge_width = 4)
    z=z+1
    print (z)
  



# to load saved graph images
    
data1=np.ones((len(hypn),64,64,3))   
labels = np.zeros((len(hypn),5))
z=0
while (z < len(hypn)):
    
            test_image1 = image.load_img(directory + '\\nonlinear\\'+str('eog_eeg') + str(z)+'.jpg', target_size = (64, 64,3))
            img1 = np.array(test_image1) 
            data1[z,:,:,3:6] = img1 /255
            
            test_image2 = image.load_img(directory + '\\PSD\\'+str('eeg') + str(z)+'.jpg', target_size = (64, 64,3))
            img2 = np.array(test_image2) 
            data1[z,:,:,0:3] = img2 /255

            labels[z,int(hypn[z])]=1
            print (z)
            z = z + 1



