import numpy as np
import cv2

class DataGenerator:
    def __init__(self, width=120, height=120, frames=30, channel=3, crop = True, normalize = False, affine = False, flip = False, edge = False  ):
        self.width = width   # X dimension of the image
        self.height = height # Y dimesnion of the image
        self.frames = frames # length/depth of the video frames
        self.channel = channel # number of channels in images 3 for color(RGB) and 1 for Gray  
        self.affine = affine # augment data with affine transform of the image
        self.flip = flip
        self.normalize =  normalize
        self.edge = edge # edge detection
        self.crop = crop

    # Helper function to generate a random affine transform on the image
    def __get_random_affine(self): # private method
        dx, dy = np.random.randint(-1.7, 1.8, 2)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return M

    # Helper function to initialize all the batch image data and labels
    def __init_batch_data(self, batch_size): # private method
        batch_data = np.zeros((batch_size, self.frames, self.width, self.height, self.channel)) 
        batch_labels = np.zeros((batch_size,5)) # batch_labels is the one hot representation of the output
        return batch_data, batch_labels

    def __load_batch_images(self, source_path, folder_list, batch_num, batch_size, t): # private method
    
        batch_data,batch_labels = self.__init_batch_data(batch_size)
        # We will also build a agumented batch data
        if self.affine:
            batch_data_aug,batch_labels_aug = self.__init_batch_data(batch_size)
        if self.flip:
            batch_data_flip,batch_labels_flip = self.__init_batch_data(batch_size)

        #create a list of image numbers you want to use for a particular video
        img_idx = [x for x in range(0, self.frames)] 

        for folder in range(batch_size): # iterate over the batch_size
            # read all the images in the folder
            imgs = sorted(os.listdir(source_path+'/'+ t[folder + (batch_num*batch_size)].split(';')[0])) 
            # Generate a random affine to be used in image transformation for buidling agumented data set
            M = self.__get_random_affine()
            
            #  Iterate over the frames/images of a folder to read them in
            for idx, item in enumerate(img_idx): 
                image = cv2.imread(source_path+'/'+ t[folder + (batch_num*batch_size)].strip().split(';')[0]+'/'+imgs[item], cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                #crop the images and resize them. Note that the images are of 2 different shape 
                #and the conv3D will throw error if the inputs in a batch have different shapes  
                image = self.crop? self.__crop(image) : image
                # If normalize is set normalize the image else use the raw image.
                resized = self.normalize? self.__normalize(self.__resize(image)) : self.__resize(image)
                # If the input is edge detected image then use the sobelx, sobely and laplacian as 3 channel of the edge detected image
                resized = self.edge? self.__edge(resized) : resized

                batch_data[folder,idx] = resized
                if self.affine:
                    batch_data_aug[folder,idx] = self.__affine(resized)   
                if self.flip:
                    batch_data_flip[folder,idx] = self.__flip(resized)   

            batch_labels[folder, int(t[folder + (batch_num*batch_size)].strip().split(';')[2])] = 1
            
            if self.affine:
                batch_labels_aug[folder, int(t[folder + (batch_num*batch_size)].strip().split(';')[2])] = 1
            
            if self.flip
                if int(t[folder + (batch*batch_size)].strip().split(';')[2])==0:
                    batch_labels_flip[folder, 1] = 1
                elif int(t[folder + (batch*batch_size)].strip().split(';')[2])==1:
                    batch_labels_flip[folder, 0] = 1
                else:
                    batch_labels_flip[folder, int(t[folder + (batch*batch_size)].strip().split(';')[2])] = 1
        
        if self.affine:
            batch_data = np.append(batch_data, batch_data_aug, axis = 0) 
            batch_labels = np.append(batch_labels, batch_labels_aug, axis = 0) 
        if self.flip:
            batch_data = np.append(batch_data, batch_data_flip, axis = 0) 
            batch_labels = np.append(batch_labels, batch_labels_flip, axis = 0) 

        return batch_data, batch_labels
    
    def generator(self, source_path, folder_list, batch_size): # public method
        print( 'Source path = ', source_path, '; batch size =', batch_size)
        while True:
            t = np.random.permutation(folder_list)
            num_batches = len(folder_list)//batch_size # calculate the number of batches
            for batch in range(num_batches): # we iterate over the number of batches
                # you yield the batch_data and the batch_labels, remember what does yield do
                yield self.__load_batch_images(source_path, folder_list, batch, batch_size, t) 
            
            # Code for the remaining data points which are left after full batches
            if (len(folder_list) != batch_size*num_batches):
                batch_size = len(folder_list) - (batch_size*num_batches)
                yield self.__load_batch_images(source_path, folder_list, num_batches, batch_size, t)

    # Helper function to perfom affice transform on the image
    def __affine(self, image, M):
        return cv2.warpAffine(image, M, (image.shape[0], image.shape[1]))

    # Helper function to flip the image
    def __flip(self, image):
        return np.flip(image,1)
    
    # Helper function to normalise the data
    def __normalize(self, image):
        return image/127.5-1
    
    # Helper function to resize the image
    def __resize(self, image):
        return cv2.resize(image, (self.width,self.height), interpolation = cv2.INTER_AREA)
    
    # Helper function to crop the image
    def __crop(self, image):
        if image.shape[0] != image.shape[1]:
            return image[0:120, 20:140]
        else:
            return image

    # Helper function for edge detection
    def __edge(self, image):
        return image





