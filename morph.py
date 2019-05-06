import cv2
import numpy as np 
import matplotlib.pyplot as plt

class Morphological_operations():
			
	def erosion(self, img, kernel):
		kern_center = (kernel.shape[0]//2,kernel.shape[1]//2)
		kernel_ones_count = kernel.sum()
		eroded_img = np.zeros((img.shape[0]+kernel.shape[0]-1, img.shape[1]+kernel.shape[1]-1))
		img_shape = img.shape
		

		x_append = np.zeros((img.shape[0],kernel.shape[1]-1))
		img = np.append(img, x_append, axis=1)
		y_append = np.zeros((kernel.shape[0]-1,img.shape[1]))
		img = np.append(img, y_append, axis=0)

		#print(kernel_ones_count)
		
		for i in range(img_shape[0]):
			for j in range(img_shape[1]):
				i_ = i+kernel.shape[0]
				j_ = j+kernel.shape[1]
				if kernel_ones_count == (kernel*img[i:i_,j:j_]).sum()/255:
					eroded_img[i+kern_center[0],j+kern_center[1]] = 1

		return(eroded_img[:img_shape[0],:img_shape[1]])

	def dilation(self, img, kernel):
		kern_center = (kernel.shape[0]//2,kernel.shape[1]//2)
		kernel_ones_count = kernel.sum()
		dilated_img = np.zeros((img.shape[0]+kernel.shape[0]-1, img.shape[1]+kernel.shape[1]-1))
		img_shape = img.shape
		

		x_append = np.zeros((img.shape[0],kernel.shape[1]-1))
		img = np.append(img, x_append, axis=1)
		y_append = np.zeros((kernel.shape[0]-1,img.shape[1]))
		img = np.append(img, y_append, axis=0)

		#print(kernel_ones_count)
		
		for i in range(img_shape[0]):
			for j in range(img_shape[1]):
				i_ = i+kernel.shape[0]
				j_ = j+kernel.shape[1]
				if (kernel*img[i:i_,j:j_]).sum() != 0:
					dilated_img[i+kern_center[0],j+kern_center[1]] = 1

		return(dilated_img[:img_shape[0],:img_shape[1]])

	def opening(self, img, kernel):
		opened_img = self.erosion(img, kernel)
		opened_img = self.dilation(opened_img, kernel)

		return(opened_img)

	def closing(self, img, kernel):
		closed_img = self.dilation(img, kernel)
		closed_img = self.erosion(closed_img, kernel)

		return(closed_img)	

def image_preprocess(path_to_image):
    image = cv2.imread(path_to_image)
    img_resize = cv2.resize(image,(512,512),interpolation = cv2.INTER_AREA)
    img_resize = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
    img_resize = cv2.equalizeHist(img_resize)
    img_resize = cv2.bitwise_not(img_resize)
    _,img_resize = cv2.threshold(img_resize,243,255,cv2.THRESH_BINARY)
    return(img_resize)

def main():
	a = Morphological_operations()

	image = image_preprocess("A.jpg")
	plt.imshow(image,cmap='Greys')
	plt.show()

	kernel = np.ones((5,5))
	image_ = a.erosion(image,kernel)
	plt.imshow(image_,cmap='Greys')
	plt.show()

	image_ = a.dilation(image,kernel)
	plt.imshow(image_,cmap='Greys')
	plt.show()

	image_ = a.closing(image,kernel)
	plt.imshow(image_,cmap='Greys')
	plt.show()
	

if __name__ == '__main__':
	main()
