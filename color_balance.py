import cv2

def change_channel(x, channel):
    for i in range(len(channel)):
        channel[i] = (channel[i]+x)//2
    return channel

def nothing(x):
    return x

title_window = "Result"
cv2.namedWindow(title_window)

cv2.createTrackbar("B", title_window, 0, 255, nothing)      # Tạo trackbar với kênh màu Blue
cv2.createTrackbar("G", title_window, 0, 255, nothing)      # Tạo trackbar với kênh màu Green
cv2.createTrackbar("R", title_window, 0, 255, nothing)      # Tạo trackbar với kênh màu Red

while(1):
    img = cv2.imread("image.jpg")
    B,G,R = cv2.split(img)

    k = cv2.waitKey(1) & 0xFF
    if k==27:
        break

    b = cv2.getTrackbarPos("B", title_window)   #Lấy giá trị từ trackbar
    g = cv2.getTrackbarPos("G", title_window)
    r = cv2.getTrackbarPos("R", title_window)

    change_channel(b, B)
    change_channel(g, G)
    change_channel(r, R)

    img = cv2.merge([B,G,R])
    img = cv2.imshow(title_window, img)
        
    #img[:] = [b,g,r]
    #img = cv2.imshow(title_window, img)

cv2.imshow(title_window, img)
cv2.waitKey(0)
cv2.destroyAllWindows() 
