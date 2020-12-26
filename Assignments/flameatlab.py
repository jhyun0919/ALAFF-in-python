import numpy as np


def get_matrix_dimension(matrix):
    """[summary]

    Args:
        matrix ([type]): [description]

    Returns:
        [type]: [description]
    """
    if type(matrix) == np.ndarray:
        m, n = matrix.shape
    else:
        m = 0
        n = 0

    return m, n


def trim_matrix_dimension(matrix, wise=None):
    """[summary]

    Args:
        matrix ([type]): [description]
        wise ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    assert wise in ["ROW", "COL", None]

    if type(matrix) == np.int64 or type(matrix) == np.float64:
        matrix = np.expand_dims(np.expand_dims(matrix, axis=0), axis=1)
    elif type(matrix) == np.ndarray:
        try:
            _, _ = matrix.shape
        except ValueError:
            if wise == "ROW":
                matrix = np.expand_dims(matrix, axis=0)
            else:
                matrix = np.expand_dims(matrix, axis=1)
    else:
        matrix = None

    return matrix


# FLA_Cont
def FLA_Cont_with_1x3_to_1x2(A0, A1, A2, side):
    """
    Purpose:
        Update the 1 x 2 partitioning of matrix A by moving the boundaries
        so that A1 is added to the side indicated by side.

    Args:
        A0 ([type]): [description]
        A1 ([type]): [description]
        A2 ([type]): [description]
        side ([type]): [description]

    Returns:
        [type]: [description]
    """
    A0 = trim_matrix_dimension(A0, "COL")
    A1 = trim_matrix_dimension(A1, "COL")
    A2 = trim_matrix_dimension(A2, "COL")

    m0, n0 = get_matrix_dimension(A0)
    m1, n1 = get_matrix_dimension(A1)
    m2, n2 = get_matrix_dimension(A2)

    # check input params
    assert m0 == m1 and m1 == m2
    assert side in ["FLA_LEFT", "FLA_RIGHT"]

    # continue with...
    if side == "FLA_LEFT":
        if m0 == 0 or n0 + n1 == 0:
            AL = np.zeros((m0, n0 + n1))
        else:
            AL = np.hstack((A0, A1))
        AR = A2
    else:
        AL = A0
        if m0 == 0 or n0 + n1 == 0:
            AR = np.zeros((m0, n1 + n2))
        else:
            AR = np.hstack((A1, A2))

    return AL, AR


def FLA_Cont_with_3x1_to_2x1(A0, A1, A2, side):
    """
    Purpose:
        Update the 2 x 1 partitioning of matrix A by moving the
        boundaries so that A1 is added to the side indicated by side.

    Args:
        A0 ([type]): [description]
        A1 ([type]): [description]
        A2 ([type]): [description]
        side ([type]): [description]

    Returns:
        [type]: [description]
    """
    A0 = trim_matrix_dimension(A0, "ROW")
    A1 = trim_matrix_dimension(A1, "ROW")
    A2 = trim_matrix_dimension(A2, "ROW")

    m0, n0 = get_matrix_dimension(A0)
    m1, n1 = get_matrix_dimension(A1)
    m2, n2 = get_matrix_dimension(A2)

    # check input params
    assert n0 == n1 and n1 == n2
    assert side in ["FLA_TOP", "FLA_BOTTOM"]

    # continue with...
    if side == "FLA_TOP":
        if m0 + m1 == 0 or n0 == 0:
            AT = np.zeros((m0 + m1, n0))
        else:
            AT = np.vstack((A0, A1))
        AB = A2
    else:
        AT = A0
        if m1 + m2 == 0 or n0 == 0:
            AB = np.zeros((m1 + m2, n0))
        else:
            AB = np.vstack((A1, A2))

    return AT, AB


def FLA_Cont_with_3x3_to_2x2(A00, A01, A02, A10, A11, A12, A20, A21, A22, quadrant):
    """
    Purpose:
        Update the 2 x 2 partitioning of matrix A by
        moving the boundaries so that A11 is added to
        the quadrant indicated by quadrant.

    Args:
        A00 ([type]): [description]
        A01 ([type]): [description]
        A02 ([type]): [description]
        A10 ([type]): [description]
        A11 ([type]): [description]
        A12 ([type]): [description]
        A20 ([type]): [description]
        A21 ([type]): [description]
        A22 ([type]): [description]
        quadrant ([type]): [description]

    Returns:
        [type]: [description]
    """
    A00 = trim_matrix_dimension(A00)
    A01 = trim_matrix_dimension(A01)
    A02 = trim_matrix_dimension(A02)
    A10 = trim_matrix_dimension(A10)
    A11 = trim_matrix_dimension(A11)
    A12 = trim_matrix_dimension(A12)
    A20 = trim_matrix_dimension(A20)
    A21 = trim_matrix_dimension(A21)
    A22 = trim_matrix_dimension(A22)

    m00, n00 = get_matrix_dimension(A00)
    m01, n01 = get_matrix_dimension(A01)
    m02, n02 = get_matrix_dimension(A02)
    m10, n10 = get_matrix_dimension(A10)
    m11, n11 = get_matrix_dimension(A11)
    m12, n12 = get_matrix_dimension(A12)
    m20, n20 = get_matrix_dimension(A20)
    m21, n21 = get_matrix_dimension(A21)
    m22, n22 = get_matrix_dimension(A22)

    # check input params
    assert m00 == m01 and m01 == m02
    assert m10 == m11 and m11 == m12
    assert m20 == m21 and m21 == m22
    assert n00 == n10 and n10 == n20
    assert n01 == n11 and n11 == n21
    assert n02 == n12 and n12 == n22
    assert quadrant in ["FLA_TL", "FLA_TR", "FLA_BL", "FLA_BR"]

    # continue with...

    if quadrant == "FLA_TL":
        # ATL
        if m00 + m10 == 0 or n00 + n01 == 0:
            ATL = np.zeros((m00 + m10, n00 + n01))
        else:
            ATL = np.vstack((np.hstack((A00, A01)), np.hstack((A10, A11))))
        # ATR
        if m02 + m12 == 0 or n02 == 0:
            ATR = np.zeros((m02 + m12, n02))
        else:
            ATR = np.vstack((A02, A12))
        # ABL
        if m20 == 0 or (n20 + n21) == 0:
            ABL = np.zeros((m20, n20 + n21))
        else:
            ABL = np.hstack((A20, A21))
        # ABR
        ABR = A22

    elif quadrant == "FLA_TR":
        # ATL
        if m00 + m10 == 0 or n00 == 0:
            ATL = np.zeros((m00 + m10, n00))
        else:
            ATL = np.vstack((A00, A10))
        # ATR
        if m01 + m11 == 0 or n01 + n02 == 0:
            ATR = np.zeros((m01 + m11, n01 + n02))
        else:
            ATR = np.vstack((np.hstack((A01, A02)), np.hstack((A11, A12))))
        # ABL
        ABL = A20
        # ABR
        if m21 == 0 or n21 + n22 == 0:
            ABR = np.zeros((m21, n21 + n22))
        else:
            ABR = np.hstack((A21, A22))

    elif quadrant == "FLA_BL":
        # ATL
        if m00 == 0 or n00 + n01 == 0:
            ATL = np.zeros((m00, n00 + n01))
        else:
            ATL = np.hstack((A00, A01))
        # ATR
        ATR = A02
        # ABL
        if m10 + m20 == 0 == 0 or n10 + n11 == 0:
            ABL = np.zeros((m10 + m20, n10 + n11))
        else:
            ABL = np.vstack((np.hstack((A10, A11)), np.hstack((A20, A21))))
        # ABR
        if m12 + m22 == 0 or n12 == 0:
            ABR = np.zeros((m12 + m22, n12))
        else:
            ABR = np.vstack((A12, A22))

    else:
        # ATL
        ATL = A00
        # ATR
        if m01 == 0 or n01 + n02 == 0:
            ATR = np.zeros((m01, n01 + n02))
        else:
            ATR = np.hstack((A01, A02))
        # ABL
        if m10 + m20 == 0 or n10 == 0:
            ABL = np.zeros((m10 + m20, n10))
        else:
            ABL = np.vstack((A10, A20))
        # ABR
        if m11 + m21 == 0 or n11 + n12 == 0:
            ABR = np.zeros((m11 + m21, n11 + n12))
        else:
            ABR = np.vstack((np.hstack((A11, A12)), np.hstack((A21, A22))))

    return ATL, ATR, ABL, ABR


# FLA_Part
def FLA_Part_1x2(A, nb, side):
    """
    Purpose:
        Partition matrix A into a left and a right side
        where the side indicated by side has nb columns.

    Args:
        A ([type]): [description]
        nb ([type]): [description]
        side ([type]): [description]

    Returns:
        [type]: [description]
    """
    A = trim_matrix_dimension(A, "COL")

    # check input params
    assert side in ["FLA_LEFT", "FLA_RIGHT"]

    # partitioning...
    if side == "FLA_LEFT":
        AL = A[:, :nb]
        AR = A[:, nb:]
    else:
        AL = A[:, :-nb]
        AR = A[:, -nb:]

    return AL, AR


def FLA_Part_2x1(A, mb, side):
    """
    Purpose:
        Partition matrix A into a top and a bottom side
        where the side indicated by side has 𝚖𝚋 rows.

    Args:
        A ([type]): [description]
        mb ([type]): [description]
        side ([type]): [description]

    Returns:
        [type]: [description]
    """
    A = trim_matrix_dimension(A, "ROW")

    # check input params
    assert side in ["FLA_TOP", "FLA_BOTTOM"]

    # partitioning...
    if side == "FLA_TOP":
        AT = A[:mb, :]
        AB = A[mb:, :]
    else:
        AT = A[:-mb, :]
        AB = A[-mb:, :]

    return AT, AB


def FLA_Part_2x2(A, mb, nb, quadrant):
    """
    Purpose:
        Partition matrix A into four quadrants
        where the quadrant indicated by quadrant is mb x nb.

    Args:
        A ([type]): [description]
        mb ([type]): [description]
        nb ([type]): [description]
        quadrant ([type]): [description]

    Returns:
        [type]: [description]
    """
    A = trim_matrix_dimension(A)

    # check input params
    assert quadrant in ["FLA_TL", "FLA_TR", "FLA_BL", "FLA_BR"]

    # partitioning...
    if quadrant == "FLA_TL":
        ATL = A[:mb, :nb]
        ATR = A[:mb, nb:]
        ABL = A[mb:, :nb]
        ABR = A[mb:, nb:]
    elif quadrant == "FLA_TR":
        ATL = A[:mb, :-nb]
        ATR = A[:mb, -nb:]
        ABL = A[mb:, :-nb]
        ABR = A[mb:, -nb:]
    elif quadrant == "FLA_BL":
        ATL = A[:-mb, :nb]
        ATR = A[:-mb, nb:]
        ABL = A[-mb:, :nb]
        ABR = A[-mb:, nb:]
    else:
        ATL = A[:-mb, :-nb]
        ATR = A[:-mb, -nb:]
        ABL = A[-mb:, :-nb]
        ABR = A[-mb:, -nb:]

    return ATL, ATR, ABL, ABR


# FLA_Repart
def FLA_Repart_1x2_to_1x3(AL, AR, nb, side):
    """
    Purpose:
        Repartition a 1 x 2 partitioning of matrix A into
        a 1 x 3 partitioning where submatrix A1 with nb columns is split from
        the side indicated by side.

    Args:
        AL ([type]): [description]
        AR ([type]): [description]
        nb ([type]): [description]
        side ([type]): [description]

    Returns:
        [type]: [description]
    """
    AL = trim_matrix_dimension(AL, "COL")
    AR = trim_matrix_dimension(AR, "COL")

    # check input params
    assert side in ["FLA_LEFT", "FLA_RIGHT"]

    # repartitioning...
    if side == "FLA_LEFT":
        A0 = AL[:, :-nb]
        A1 = AL[:, -nb:]
        A2 = AR
    else:
        A0 = AL
        A1 = AR[:, :nb]
        A2 = AR[:, nb:]

    return A0, A1, A2


def FLA_Repart_2x1_to_3x1(AT, AB, mb, side):
    """
    Purpose:
        Repartition a 2 x 1 partitioning of matrix A into
        a 3 x 1 partitioning where submatrix A1 with mb rows is split from
        the side indicated by side.

    Args:
        AT ([type]): [description]
        AB ([type]): [description]
        mb ([type]): [description]
        side ([type]): [description]

    Returns:
        [type]: [description]
    """
    AT = trim_matrix_dimension(AT, "ROW")
    AB = trim_matrix_dimension(AB, "ROW")

    # check input params
    assert side in ["FLA_TOP", "FLA_BOTTOM"]

    # repartitioning...
    if side == "FLA_TOP":
        A0 = AT[:-mb, :]
        A1 = AT[-mb:, :]
        A2 = AB
    else:
        A0 = AT
        A1 = AB[:mb, :]
        A2 = AB[mb:, :]

    return A0, A1, A2


def FLA_Repart_2x2_to_3x3(ATL, ATR, ABL, ABR, bm, bn, quadrant):
    """
    Purpose:
        Repartition a 2 x 2 partitioning of matrix A into
        a 3 x 3 partitioning where mb x nb submatrix A11 is split from
        the quadrant indicated by quadrant.

    Args:
        ATL ([type]): [description]
        ATR ([type]): [description]
        ABL ([type]): [description]
        ABR ([type]): [description]
        bm ([type]): [description]
        bn ([type]): [description]
        quadrant ([type]): [description]

    Returns:
        [type]: [description]
    """
    ATL = trim_matrix_dimension(ATL)
    ATR = trim_matrix_dimension(ATR)
    ABL = trim_matrix_dimension(ABL)
    ABR = trim_matrix_dimension(ABR)

    # check input params
    assert quadrant in ["FLA_TL", "FLA_TR", "FLA_BL", "FLA_BR"]

    # repartitioning...
    if quadrant == "FLA_TL":
        A00 = ATL[:-bm, :-bn]
        A01 = ATL[:-bm, -bn:]
        A02 = ATR[:-bm, :]
        A10 = ATL[-bm:, :-bn]
        A11 = ATL[-bm:, -bn:]
        A12 = ATR[-bm:, :]
        A20 = ABL[:, :-bn]
        A21 = ABL[:, -bn:]
        A22 = ABR
    elif quadrant == "FLA_TR":
        A00 = ATL[:-bm, :]
        A01 = ATR[:-bm, :bn]
        A02 = ATR[:-bm, bn:]
        A10 = ATL[-bm:, :]
        A11 = ATR[-bm:, :bn]
        A12 = ATR[-bm:, bn:]
        A20 = ABL
        A21 = ABR[:, :bn]
        A22 = ABR[:, bn:]
    elif quadrant == "FLA_BL":
        A00 = ATL[:, :-bn]
        A01 = ATL[:, -bn:]
        A02 = ATR
        A10 = ABL[:bm, :-bn]
        A11 = ABL[:bm, -bn:]
        A12 = ABR[:bm, :]
        A20 = ABL[bm:, :-bn]
        A21 = ABL[bm:, -bn:]
        A22 = ABR[bm:, :]
    else:
        A00 = ATL
        A01 = ATR[:, :bn]
        A02 = ATR[:, bn:]
        A10 = ABL[:bm, :]
        A11 = ABR[:bm, :bn]
        A12 = ABR[:bm, bn:]
        A20 = ABL[bm:, :]
        A21 = ABR[bm:, :bn]
        A22 = ABR[bm:, bn:]

    return A00, A01, A02, A10, A11, A12, A20, A21, A22
