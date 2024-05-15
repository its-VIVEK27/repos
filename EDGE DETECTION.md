Hi, 
firstly it was very fun to work on the first task of the summer project and I enjoyed it thoroughly.
How I approached the task was to apply everything I know first, then I learnt various method which I'll explain and tried to find edges in the image.

**Steps-**
1. Importing the libraries and reading the images
2. first i used sobel directly from the function but found we had to the convolution by writing it ourself and it was more interesting.
3. writing a function for performing **convolution**
4. defining filters such as sobel and scharr to find gradient 
5. writing function to perform blurring and sharpening

**Experimentation-**
1.tried sobel on blurred image, sharp image, blurred sharp image and on sharpened blur image, the best result I got is sobel on blurred sharp image. other were also good.
2. Tried scharr on blurred image, sharp image, blurred sharp image and on sharpened blur image. Scharr results were quite disappointing and did not detect the edge as good as sobel did.
3. I also tried using a 5x5 kernel instead of 3x3 kernel to perform gaussian blur, the results were similar and didn't have great difference.

**canny edge detection**
1. read the maths behind canny edge detection and how it is better than other methods.
2. got to know that the basic steps are - 
- noise reduction
- non maximum suppression: at each pixel , checking in the direction of gradient whether pixel in that direction are of greater intensity or not, if greater, that pixel is also converted to brighter one.
- double thresholding: classifying a pixel as either strong or weak, tweaked the low thershold and high threshold ratio but got similar results only.
- hystersis: if a pixel is weak but all the pixel around it are strong then the pixel is converted to strong.
3.implemented these step in my code 
4. the result were okaish and could have been improved by tweaking some setting.


