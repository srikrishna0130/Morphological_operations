import cv2  # OpenCV library for image processing
import numpy as np  # NumPy library for numerical operations
import matplotlib.pyplot as plt  # Matplotlib library for plotting

# Class for performing morphological operations
class Morphological_operations():
    # The erosion operation
    def erosion(self, img, kernel):
        # Calculate the center of the kernel
        kern_center = (kernel.shape[0]//2,kernel.shape[1]//2)
        
        # Count the number of ones in the kernel
        kernel_ones_count = kernel.sum()
        
        # Initialize the eroded image with zeros
        eroded_img = np.zeros((img.shape[0]+kernel.shape[0]-1, img.shape[1]+kernel.shape[1]-1))
        
        # Save the shape of the original image
        img_shape = img.shape
        
        # Append zeros to the image to match the size of the kernel
        x_append = np.zeros((img.shape[0],kernel.shape[1]-1))
        img = np.append(img, x_append, axis=1)
        
        y_append = np.zeros((kernel.shape[0]-1,img.shape[1]))
        img = np.append(img, y_append, axis=0)
        
        # Perform the erosion operation
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                i_ = i+kernel.shape[0]
                j_ = j+kernel.shape[1]
                
                # If the sum of the kernel and the image slice is equal to the number of ones in the kernel, set the pixel in the eroded image to 1
                if kernel_ones_count == (kernel*img[i:i_,j:j_]).sum()/255:
                    eroded_img[i+kern_center[0],j+kern_center[1]] = 1

        # Return the eroded image
        return(eroded_img[:img_shape[0],:img_shape[1]])

    # The dilation operation
    def dilation(self, img, kernel):
        # Calculate the center of the kernel
        kern_center = (kernel.shape[0]//2,kernel.shape[1]//2)
        
        # Count the number of ones in the kernel
        kernel_ones_count = kernel.sum()
        
        # Initialize the dilated image with zeros
        dilated_img = np.zeros((img.shape[0]+kernel.shape[0]-1, img.shape[1]+kernel.shape[1]-1))
        
        # Save the shape of the original image
        img_shape = img.shape
        
        # Append zeros to the image to match the size of the kernel
        x_append = np.zeros((img.shape[0],kernel.shape[1]-1))
        img = np.append(img, x_append, axis=1)
        
        y_append = np.zeros((kernel.shape[0]-1,img.shape[1]))
        img = np.append(img, y_append, axis=0)
        
        # Perform the dilation operation
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                i_ = i+kernel.shape[0]
                j_ = j+kernel.shape[1]
                
                # If the sum of the kernel and the image slice is not zero, set the pixel in the dilated image to 1
                if (kernel*img[i:i_,j:j_]).sum() != 0:
                    dilated_img[i+kern_center[0],j+kern_center[1]] = 1

        # Return the dilated image
        return(dilated_img[:img_shape[0],:img_shape[1]])

    # The opening operation
    def opening(self, img, kernel):
        # Perform erosion followed by dilation
        opened_img = self.erosion(img, kernel)
        opened_img = self.dilation(opened_img, kernel)

        # Return the opened image
        return(opened_img)

    # The closing operation
    def closing(self, img, kernel):
        # Perform dilation followed by erosion
        closed_img = self.dilation(img, kernel)
        closed_img = self.erosion(closed_img, kernel)

        # Return the closed image
        return(closed_img)	

# Function to preprocess the image
def image_preprocess(path_to_image):
    # Read the image
    image = cv2.imread(path_to_image)
    
    # Resize the image
    img_resize = cv2.resize(image,(512,512),interpolation = cv2.INTER_AREA)
    
    # Convert the image to grayscale
    img_resize = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
    
    # Equalize the histogram of the image
    img_resize = cv2.equalizeHist(img_resize)
    
    # Invert the image
    img_resize = cv2.bitwise_not(img_resize)
    
    # Threshold the image
    _,img_resize = cv2.threshold(img_resize,243,255,cv2.THRESH_BINARY)
    
    # Return the preprocessed image
    return(img_resize)

# Main function
def main():
    # Create an object of the Morphological_operations class
    a = Morphological_operations()

    # Preprocess the image
    image = image_preprocess("A.jpg")
    
    # Display the image
    plt.imshow(image,cmap='Greys')
    plt.show()

    # Create a 5x5 kernel
    kernel = np.ones((5,5))
    
    # Perform and display the erosion operation
    image_ = a.erosion(image,kernel)
    plt.imshow(image_,cmap='Greys')
    plt.show()

    # Perform and display the dilation operation
    image_ = a.dilation(image,kernel)
    plt.imshow(image_,cmap='Greys')
    plt.show()

    # Perform and display the closing operation
    image_ = a.closing(image,kernel)
    plt.imshow(image_,cmap='Greys')
    plt.show()

# If this script is run directly, call the main function
if __name__ == '__main__':
    main()