import numpy as np

def get_distance(p1,p2):
    """ calculates the difference between 2 points

    Args:
        p1 ([x,y])
        p2 ([x,y])

    Return:
        distance according to the Pythagoras theorem
    """
    x = p1[0] + p2[0]
    y = p1[1] + p2[0]
    return np.sqrt(x**2 + y**2)

def get_pw_matrix():
    print()

def line(prev,curr,next,val):
    """ checks if the landmarks are in a line
    """
    ret = False
    if val is 0:
        if np.array_equal(prev.get_x(),next.get_x()):
            if np.array_equal(prev.get_x(),curr.get_x()):
                ret = True
    elif val is 1:
        if np.array_equal(prev.get_y(),next.get_y()):
            if np.array_equal(prev.get_y(),curr.get_y()):
                ret = True
    return ret

def angle(landmark1,landmark2):
    """gets the angle between 2 landmarks
    """
    x = np.mean([landmark1.get_x(),landmark2.get_x()])
    y = np.mean([landmark1.get_y(),landmark2.get_y()])
    return [x,y]

def subtract(landmark1,landmark2):
    """ subtracts two landmarks
    """
    x = landmark1.get_x()-landmark2.get_x()
    y = landmark1.get_y()-landmark2.get_y()
    return [x,y]

def subtract_angle(landmark,angle):
    """ private method
    subtracts a landmark from an angle
    """
    x = landmark.get_x() - angle[0]
    y = landmark.get_y() - angle[1]
    return [x,y]

def normalized(prev,curr,next):
    angle = angle(next,prev)
    d = subtract_angle(curr,angle)
    if np.array_equal(d,[0,0]):
        d = subtract(next,prev)
    mag = np.sqrt(d[0]**2,d[1]**2)
    return d/mag
