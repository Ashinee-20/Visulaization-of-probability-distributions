from scipy.integrate import quad
from scipy.stats import norm, uniform, poisson
from numpy import exp, pi, inf
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def NormExpoeval(z):
    return exp(-(z ** 2) / 2)

def pointsExpoeval(l):
    return exp(-l)

def normalDist():
    m = int(input("Enter the mean value: "))
    s = int(input("Enter the standard deviation value: "))
    x = int(input("Enter the value: "))

    print(f"Mean value is: {m}")
    print(f"Standard Deviation value is: {s}")
    z_val = (x - m) / s
    result = quad(NormExpoeval, z_val, inf)
    # plotting pdf for normal distribution
    x_axis = np.arange(-2 * m, 2 * m, 0.01)
    plt.title('PDF of Normal Distribution with CDF highlighted in blue region', fontsize='10')
    plt.xlabel('Length', fontsize='10')
    plt.ylabel('Probability', fontsize='10')
    plt.plot(x_axis, norm.pdf(x_axis, m, s))
    plt.fill_between(x=np.arange(x, 2 * m, 0.01), y1=norm.pdf(np.arange(x, 2 * m, 0.01), m, s), facecolor="blue")
    plt.show()

    return (result[0]/(math.sqrt(2*pi)))


def uniformDist():
    a = int(input("Enter the lower limit: "))
    b = int(input("Enter the upper limit: "))
    m = (a + b) / 2
    s = ((b - a) ** 2) / 12
    print(f"Mean value is: {m}")
    print(f"Standard Deviation value is: {s}")
    result = 1 / (b - a)

    size = 1000
    # for plotting the cdf graph
    x = np.linspace(uniform.ppf(0.01, a, b), uniform.ppf(0.99, a, b), 100)
    y = uniform.cdf(x, a, b)
    plt.plot(x, y, 'b-', label='CDF')
    plt.legend(loc='upper left')
    plt.show()

    return result


def poissonDist():
    l = int(input("Enter the lambda value: "))
    k = int(input("Enter the k value: "))
    print(f"Mean value is: {l}")
    print(f"Standard Deviation value is: {l}")
    res = (pointsExpoeval(l) * (l ** k)) / (math.factorial(k))

    # for plotting cdf graph
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.35)
    x = np.arange(0, 20, 0.1)
    y = poisson.cdf(x, l)
    ma, = plt.plot(x, y)
    plt.title("PMF graph of Poisson distribution ")
    plt.xlabel("Random variable")
    plt.ylabel("Probability")
    axlambda = plt.axes([0.25, 0.15, 0.65, 0.03])
    l = Slider(axlambda, 'Lambda', 0.0, 10.0, l)

    def update(val):
        lam = l.val
        ma.set_ydata(poisson.cdf(x, lam))

    l.on_changed(update)
    plt.show()

    return res



print("Enter the corresponding number for the type of distribution that you would like to view:\n 1-Normal Distribution\n 2-Uniform Distribution\n 3-Possion Distribution")
print("\n")
n = 0

while(n!=4):
    n = int(input("Enter the number: "))
    if(n==1):
        res1 = normalDist()
        print("The cdf for normal distribution: ", res1)
        print("\n")
    if(n==2):
        res2 = uniformDist()
        print("The cdf for Uniform distribution: ", res2)
        print("\n")
    if(n==3):
        res3 = poissonDist()
        print("The cdf for Poisson distribution: ", res3)
        print("\n")