import numpy as np
from landmarks import Landmarks

def procrustes(landmarks):
    """Applies Procrustes Analysis to landmark models based on An Introduction to Active Shape Models Protocal 4: Aligning a Set of Shapes and Active Shape Models-Their Training and Application Section 3.2

    Args:
        landmarks ([Landmarks]) : all landmarks for an incisors

    Returns:

    """
    aligned = list(landmarks)

    # 1 Translate each example so that its centre of gravity is at the origin.
    # aligned = [shape.translate_to_origin() for shape in aligned]

    # 2 Choose one examplle as an initial estimate of the mean shape and scale so that |x0| = 1.

    # 3 Define default orientation.

    # 4 Align all the shapes with the current estimate of the mean shape.

    # 5 Re-estimate the mean from the aligned shapes

    # 6 Apply the constrants and scale and orientation to the current estimate of the mean by aligning it with |x0| and scaling so that |x| = 1

    #7 If not converged return to 4. Convergence is clared if the estimate of the mean does not change significantly after an iteration)
