import numpy as np
import matplotlib.pyplot as plt

def old_model(t, S, I, R, b=0.3, g=0.05):

    Slist, Ilist, Rlist = [S], [I], [R]

    for i in range(len(t)-1):
        if I > 1e-6:
            Sd = -b * I * S
            Id = b * I * S - g * I
            Rd = g * I

            S_new = S + Sd * (t[i+1] - t[i])
            I_new = I + Id * (t[i+1] - t[i])
            R_new = R + Rd * (t[i+1] - t[i])

            S, I, R = S_new, I_new, R_new

        Slist.append(S)
        Ilist.append(I)
        Rlist.append(R)

    return (np.array(Slist), np.array(Ilist), np.array(Rlist))

def alpha_model(t, S, I, R, b=0.3, g=0.05, alpha=1):

    Slist, Ilist, Rlist = [S], [I], [R]

    for i in range(len(t)-1):
        if I > 1e-6:
            Sd = -b * I * S ** alpha
            Id = b * I * S ** alpha - g * I
            Rd = g * I

            S_new = S + Sd * (t[i+1] - t[i])
            I_new = I + Id * (t[i+1] - t[i])
            R_new = R + Rd * (t[i+1] - t[i])

            S, I, R = S_new, I_new, R_new

        Slist.append(S)
        Ilist.append(I)
        Rlist.append(R)

    return (np.array(Slist), np.array(Ilist), np.array(Rlist))



def new_model(t, S, I, R, b=0.3, g=0.05):

    k_max = len(S)

    k = np.arange(k_max)

    n = S + I + R
    
    k_mean = (n*k).sum()

    Slist, Ilist, Rlist = [S], [I], [R]

    for i in range(len(t)-1):
        if I.sum() > 1e-6:
            Sd = -k / k_mean * b * (n*k*I).sum() / k_mean * S
            Id = k / k_mean * b * (n*k*I).sum() / k_mean * S - g * I
            Rd = g * I
        
            S_new = S + Sd * (t[i+1] - t[i])
            I_new = I + Id * (t[i+1] - t[i])
            R_new = R + Rd * (t[i+1] - t[i])
        
            S, I, R = S_new, I_new, R_new

        Slist.append(S)
        Ilist.append(I)
        Rlist.append(R)

    return (np.array(Slist), np.array(Ilist), np.array(Rlist))
