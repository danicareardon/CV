import numpy as np
from landmarks import Landmarks

def procrustes(landmarks):
    """Applies Procrustes Analysis to landmark models based on An Introduction to Active Shape Models Protocal 4: Aligning a Set of Shapes and Active Shape Models-Their Training and Application Section 3.2

    Args:
        landmarks (list[Landmarks]) : all landmarks for an incisors

    Returns:
        mean_shape ([Landmarks]) : estimate of the mean_shape
        aligned (List[Landmarks]) : all landmarks aligned to the mean

    """
    aligned = list(landmarks)

    # 1 Translate each example so that its centre of gravity is at the origin.
    aligned = [shape.translate_to_origin() for shape in aligned]

    # 2 Choose one example as an initial estimate of the mean shape and scale so that |x0| = 1.
    # 3 Define default orientation.
    x0 = aligned[0].scale_to_one()
    mean_shape = x0

    # 4 Align all the shapes with the current estimate of the mean shape.
    while True:
        for i, shape in enumerate(aligned):
            aligned[i] = align_two_shapes(shape,mean_shape)


        # 5 Re-estimate the mean from the aligned shapes
        temp_mean_shape = get_new_mean_shape(aligned)

        # 6 Apply the constrants and scale and orientation to the current estimate of the mean by aligning it with |x0| and scaling so that |x| = 1
        temp_mean_shape = align_two_shapes(temp_mean_shape,x0)
        temp_mean_shape = temp_mean_shape.scale_to_one().translate_to_origin()

        #7 If not converged return to 4. Convergence is declared if the estimate of the mean does not change significantly after an iteration)
        if ((mean_shape.get_vector() - temp_mean_shape.get_vector()) < 1e-10).all():
            break
        mean_shape = temp_mean_shape
    return mean_shape, aligned

def get_new_mean_shape(landmarks):
    """Gets the updated mean shape

    Args:
        landmarks list[Landmarks] : list of landmark objects

    Returns:
        A new mean shape as a [Landmarks]
    """
    m = []
    for landmark in landmarks:
        m.append(landmark.get_vector())
    m = np.array(m)
    return Landmarks(np.mean(m, axis=0))


def align_two_shapes(shape1,shape2):
    """ Based off of Appendix D of An Introduction to Active Shape Models.
    Aligns two Shapes

    Args:
        shape1([Landmarks]) : the shape that will be rotated and scaled
        shape2([Landmarks]) : the shape that will be aligned

    Returns:
        [Landmarks] : shape 1 aligned
    """
    s, theta = get_s_and_theta(shape1,shape2)

    shape1 = shape1.rotate(theta)
    shape1 = shape1.scale(s)

    scaled = np.dot(shape1.get_vector(), shape2.get_vector())
    return Landmarks(shape1.get_vector()*(1.0/scaled))


def get_s_and_theta(shape1,shape2):
    """ Based off of Appendix D of An Introduction to Active Shape Models.
    Aligns two Shapes

    Args:
        shape1([Landmarks]) : the shape that will be rotated and scaled
        shape2([Landmarks]) : the shape that will be aligned

    Returns:
        s, theta: the scaling and rotating value
    """

    shape1 = shape1.get_vector()
    shape2 = shape2.get_vector()

    len1 = int(len(shape1)/2)
    len2 = int(len(shape2)/2)


    a = np.dot(shape1,shape2) / (np.linalg.norm(shape1)**2)
    b = (np.dot(shape1[:len1], shape2[len2:]) - np.dot(shape1[len1:], shape2[:len2])) / (np.linalg.norm(shape1)**2)

    s = np.sqrt(a**2+b**2)

    theta = np.arctan(b/a)

    return s,theta
