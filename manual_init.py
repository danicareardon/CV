import cv2
import radiograph as rg
global mouseX,mouseY

def mouse_click(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 600)
    cv2.imshow('image',img)
    cv2.setMouseCallback('image',__click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def __click(event,x,y,flags,param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseX = x
        mouseY = y
        print(x,y)


if __name__ == "__main__":
    imgs = rg.load_radiographs(1,2)
    for i in imgs:
        i.preprocess()
        x = i.sobel
        mouse_click(x)