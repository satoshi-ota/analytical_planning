import numpy as np
from utils import util

# def analytical_type1(v0, a0, j_dec, j_acc, t_dec, t_min, t_acc):

#     a1, v1, x1 = util.integral(a0, v0, 0.0, j_dec, t_dec)
#     a2, v2, x2 = util.integral(a1, v1, x1, 0.0, t_min)
#     a3, v3, x3 = util.integral(a2, v2, x2, j_acc, t_acc)

#     return a3, v3, x3

# def analytical_type2(j_dec, j_acc, t_dec, t_min, t_acc):

#     a1, v1, x1 = util.integral(a0, v0, x0, j_dec, t_dec)
#     a2, v2, x2 = util.integral(a2, v2, x2, j_acc, t_acc)

#     return a3, v3, x3
