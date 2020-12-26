import numpy as np
from math import sqrt
import sys

from numpy.core.fromnumeric import transpose

sys.path.append("../")
from flameatlab import *


def CGS_QR(A):
    A = trim_matrix_dimension(A)
    m, n = get_matrix_dimension(A)
    Q = np.zeros(shape=(m, n))
    R = np.zeros(shape=(n, n))

    AL, AR = FLA_Part_1x2(A, 0, "FLA_LEFT")
    QL, QR = FLA_Part_1x2(Q, 0, "FLA_LEFT")
    RTL, RTR, RBL, RBR = FLA_Part_2x2(R, 0, 0, "FLA_TL")

    while AL.shape[1] < A.shape[1]:
        A0, a1, A2 = FLA_Repart_1x2_to_1x3(AL, AR, 1, "FLA_RIGHT")
        Q0, q1, Q2 = FLA_Repart_1x2_to_1x3(QL, QR, 1, "FLA_RIGHT")

        R00, r01, R02, r10t, rho11, r12t, R20, r21, R22 = FLA_Repart_2x2_to_3x3(
            RTL, RTR, RBL, RBR, 1, 1, "FLA_BR"
        )

        # ------------------------------------------------------ #
        r01 = np.dot(np.transpose(Q0), a1)
        a1perp = a1 - np.dot(Q0, r01)
        rho11 = np.linalg.norm(a1perp)
        q1 = a1perp / rho11
        # ------------------------------------------------------ #

        AL, AR = FLA_Cont_with_1x3_to_1x2(A0, a1, A2, "FLA_LEFT")
        QL, QR = FLA_Cont_with_1x3_to_1x2(Q0, q1, Q2, "FLA_LEFT")
        RTL, RTR, RBL, RBR = FLA_Cont_with_3x3_to_2x2(
            R00, r01, R02, r10t, rho11, r12t, R20, r21, R22, "FLA_TL"
        )
    Q = np.hstack((QL, QR))
    R = np.vstack((np.hstack((RTL, RTR)), np.hstack((RBL, RBR))))

    return Q, R


def MGS_QR(A):
    A = trim_matrix_dimension(A)
    m, n = get_matrix_dimension(A)
    R = np.zeros(shape=(n, n))

    AL, AR = FLA_Part_1x2(A, 0, "FLA_LEFT")
    RTL, RTR, RBL, RBR = FLA_Part_2x2(R, 0, 0, "FLA_TL")

    while AL.shape[1] < A.shape[1]:
        A0, a1, A2 = FLA_Repart_1x2_to_1x3(AL, AR, 1, "FLA_RIGHT")

        R00, r01, R02, r10t, rho11, r12t, R20, r21, R22 = FLA_Repart_2x2_to_3x3(
            RTL, RTR, RBL, RBR, 1, 1, "FLA_BR"
        )

        # ------------------------------------------------------ #
        rho11 = np.linalg.norm(a1)
        a1 = a1 / rho11
        r12t = np.dot(np.transpose(a1), A2)
        A2 = A2 - np.dot(a1, r12t)
        # ------------------------------------------------------ #

        AL, AR = FLA_Cont_with_1x3_to_1x2(A0, a1, A2, "FLA_LEFT")
        RTL, RTR, RBL, RBR = FLA_Cont_with_3x3_to_2x2(
            R00, r01, R02, r10t, rho11, r12t, R20, r21, R22, "FLA_TL"
        )
    Q = np.hstack((AL, AR))
    R = np.vstack((np.hstack((RTL, RTR)), np.hstack((RBL, RBR))))

    return Q, R


def Housev(chi1, x2):
    """[summary]

    Args:
        chi1 ([type]): [description]
        x2 ([type]): [description]

    Returns:
        [type]: [description]
    """
    normx = sqrt(np.dot(chi1, chi1) + np.dot(np.transpose(x2), x2))

    rho = -np.sign(chi1) * normx
    nu1 = chi1 - rho
    u2 = x2 / nu1

    tau = (1 + np.dot(np.transpose(u2), u2)) / 2

    return rho, u2, tau


def Housev_alt(chi1, x2):
    """[summary]

    Args:
        chi1 ([type]): [description]
        x2 ([type]): [description]

    Returns:
        [type]: [description]
    """
    chi2 = np.linalg.norm(x2)
    alpha = np.linalg.norm(np.vstack((chi1, chi2)))
    rho = -np.sign(chi1) * alpha
    nu1 = chi1 - rho
    u2 = x2 / nu1
    chi2 = chi2 / abs(nu1)

    tau = (1 + np.dot(chi2, chi2)) / 2
    return rho, u2, tau


def HQR(A):
    """[summary]

    Args:
        A ([type]): [description]

    Returns:
        [type]: [description]
    """
    A = trim_matrix_dimension(A)
    _, n = get_matrix_dimension(A)
    t = np.zeros(shape=(n, 1))

    ATL, ATR, ABL, ABR = FLA_Part_2x2(A, 0, 0, "FLA_TL")
    tT, tB = FLA_Part_2x1(t, 0, "FLA_TOP")

    while ATL.shape[1] < A.shape[1]:
        A00, a01, A02, a10t, alpha11, a12t, A20, a21, A22 = FLA_Repart_2x2_to_3x3(
            ATL, ATR, ABL, ABR, 1, 1, "FLA_BR"
        )
        t0, tau1, t2 = FLA_Repart_2x1_to_3x1(tT, tB, 1, "FLA_BOTTOM")

        # ------------------------------------------------------ #
        alpha11, a21, tau1 = Housev(alpha11, a21)
        w12t = (a12t + np.dot(np.transpose(a21), A22)) / tau1
        a12t = a12t - w12t
        A22 = A22 - np.dot(a21, w12t)
        # ------------------------------------------------------ #

        ATL, ATR, ABL, ABR = FLA_Cont_with_3x3_to_2x2(
            A00, a01, A02, a10t, alpha11, a12t, A20, a21, A22, "FLA_TL"
        )
        tT, tB = FLA_Cont_with_3x1_to_2x1(t0, tau1, t2, "FLA_TOP")

    A_out = np.vstack((np.hstack((ATL, ATR)), np.hstack((ABL, ABR))))
    t_out = np.vstack((tT, tB))

    return A_out, t_out


def HLQ(A):
    """[summary]

    Args:
        A ([type]): [description]

    Returns:
        [type]: [description]
    """
    A = trim_matrix_dimension(A)
    m, _ = get_matrix_dimension(A)
    t = np.zeros(shape=(m, 1))

    ATL, ATR, ABL, ABR = FLA_Part_2x2(A, 0, 0, "FLA_TL")
    tT, tB = FLA_Part_2x1(t, 0, "FLA_TOP")

    while ATL.shape[0] < A.shape[0]:
        A00, a01, A02, a10t, alpha11, a12t, A20, a21, A22 = FLA_Repart_2x2_to_3x3(
            ATL, ATR, ABL, ABR, 1, 1, "FLA_BR"
        )
        t0, tau1, t2 = FLA_Repart_2x1_to_3x1(tT, tB, 1, "FLA_BOTTOM")

        # ------------------------------------------------------ #
        alpha11, u2, tau1 = Housev(alpha11, np.transpose(a12t))
        a12t = np.transpose(u2)
        w21 = (a21 + np.dot(A22, np.transpose(a12t))) / tau1

        a21 = a21 - w21
        A22 = A22 - w21 * a12t
        # ------------------------------------------------------ #

        ATL, ATR, ABL, ABR = FLA_Cont_with_3x3_to_2x2(
            A00, a01, A02, a10t, alpha11, a12t, A20, a21, A22, "FLA_TL"
        )
        tT, tB = FLA_Cont_with_3x1_to_2x1(t0, tau1, t2, "FLA_TOP")

    A_out = np.vstack((np.hstack((ATL, ATR)), np.hstack((ABL, ABR))))
    t_out = np.vstack((tT, tB))

    return A_out, t_out


def FormQ(A, t):
    """[summary]

    Args:
        A ([type]): [description]
        t ([type]): [description]

    Returns:
        [type]: [description]
    """
    A = trim_matrix_dimension(A)
    _, n = get_matrix_dimension(A)

    ATL, ATR, ABL, ABR = FLA_Part_2x2(A, n, n, "FLA_TL")
    tT, tB = FLA_Part_2x1(t, n, "FLA_TOP")

    while ABR.shape[0] < A.shape[0]:
        A00, a01, A02, a10t, alpha11, a12t, A20, a21, A22 = FLA_Repart_2x2_to_3x3(
            ATL, ATR, ABL, ABR, 1, 1, "FLA_TL"
        )
        t0, tau1, t2 = FLA_Repart_2x1_to_3x1(tT, tB, 1, "FLA_TOP")

        # ------------------------------------------------------ #
        alpha11 = 1 - 1 / tau1
        a12t = -np.dot(np.transpose(a21), A22) / tau1
        A22 = A22 + np.dot(a21, a12t)
        a21 = -a21 / tau1
        # ------------------------------------------------------ #

        ATL, ATR, ABL, ABR = FLA_Cont_with_3x3_to_2x2(
            A00, a01, A02, a10t, alpha11, a12t, A20, a21, A22, "FLA_BR"
        )
        tT, tB = FLA_Cont_with_3x1_to_2x1(t0, tau1, t2, "FLA_BOTTOM")

    A_out = np.vstack((np.hstack((ATL, ATR)), np.hstack((ABL, ABR))))

    return A_out
