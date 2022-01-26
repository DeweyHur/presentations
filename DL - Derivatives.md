---
marp: true
theme: gaia
---

<style>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
</style>

<!-- class: lead -->

# Derivatives
---

<!-- class: none -->

I suppose you're all used to the derivatives except logarithm what I am. But you will want to remind to develop the:
* Euler's number
$$ e $$
* Logarithm
$$ \log $$

---

## Euler's number

![bg right](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Hyperbola_E.svg/1920px-Hyperbola_E.svg.png)

e is the value satisfying the below equation:
$$ \lim_{h\to0}\frac{a^h-1}{h} = 1 $$

---

## Euler's number properties

$$ e = \lim_{n \to \infty} \frac{1}{1+n} $$
$$ = \sum_{n=0}^{\infty}\frac{1}{n!} $$
$$ = 1+\frac{1}{1}+\frac{1}{1\cdot2}+\frac{1}{1\cdot2\cdot3}+\cdots $$

---
## Euler's number equations

$$ e^xe^y = e^{x+y} $$
$$ \frac{e^x}{e^y} = e^{x-y} $$
$$ (e^x)^y = e^{xy} $$
$$ e^0 = 1 $$
---

## Logarithm

The inverse function to exponentiation

![bg right](https://upload.wikimedia.org/wikipedia/commons/8/81/Logarithm_plots.png)

---

## Logarithm properties

$$\log{xy} = \log{x}+\log{y}$$
$$\log(\frac{x}{y}) = \log{x}-\log{y}$$
$$\log{x^y}=y\log{x}$$

---

## Natural logarithms

Using e as a base for the logarithm

$$ \log_{e}{x} = \log{x} = \ln{x} $$

---

## Exponent rule for Derivatives
$$\frac{d}{dx}e^x=e^x$$
$$\frac{d}{dx}e^{ax}=ae^{ax}$$
$$\frac{d}{dx}(a^x)=a^x\ln{a}$$

---

## Logarithms rule for Derivatives

$$ \frac{d}{dx}\log_{a}{x}=\frac{1}{x\ln{a}} $$
$$ \frac{d}{dx}\log_{e}{x}=\frac{d}{dx}\ln{x}=\frac{1}{n} $$

---

<!-- class: lead -->

# Computation Graph

---

<!-- class: none -->

# Computation Graph

- A directed graph where the nodes correspond to mathematical operations. 
- A way of expressing and evaluating a mathematical expression.

___

$$ p=x+y $$

![width:400 center](https://www.tutorialspoint.com/python_deep_learning/images/computational_graph_equation1.jpg)

$$ g=(x+y)*z $$

![width:500 center](https://www.tutorialspoint.com/python_deep_learning/images/computational_graph_equation2.jpg)

---

## Forward Pass

$$ x=1, y=3, z=-3 $$

![width:500 center](https://www.tutorialspoint.com/python_deep_learning/images/forward_pass_equation.jpg)

$$ \therefore p=4, g=-12 $$

---

## Backward Pass
- Backpropagation
- compute the gradients for each input with respect to the final output. 
- These gradients are essential for training the neural network using gradient descent.

---

![width:500 center](https://www.tutorialspoint.com/python_deep_learning/images/backward_pass.jpg)

$$ \frac{\sigma{g}}{\sigma{g}}=1 $$
$$ g=p\cdot{z} \to \frac{\sigma{g}}{\sigma{z}}=p=4 $$
$$ g=z\cdot{p} \to \frac{\sigma{g}}{\sigma{p}}=z=-3 $$

---

![width:500 center](https://www.tutorialspoint.com/python_deep_learning/images/backward_pass.jpg)

$$ g=zx+zy \to \frac{\sigma{g}}{\sigma{x}}=z=-3 $$
$$ g=zy+zx \to \frac{\sigma{g}}{\sigma{y}}=z=-3 $$

---

## Chain Rule (Alternative way)

$$ p=x+y \to \frac{\sigma{p}}{\sigma{x}}=1, \frac{\sigma{p}}{\sigma{y}}=1 $$
$$ \frac{\sigma{g}}{\sigma{x}} = \frac{\sigma{g}}{\sigma{p}}\cdot\frac{\sigma{p}}{\sigma{x}}=z\cdot1 = -3 $$
$$ \frac{\sigma{g}}{\sigma{y}} = \frac{\sigma{g}}{\sigma{p}}\cdot\frac{\sigma{p}}{\sigma{y}}=z\cdot1 = -3 $$

<!-- _footer: Chain Rule - https://en.wikipedia.org/wiki/Chain_rule -->

---

# Where to use

![bg left](https://github.com/tensorflow/docs/blob/master/site/en/guide/images/intro_to_graphs/two-layer-network.png?raw=1)

- A computation is described using the Data Flow Graph for Tensorflow
  - Node: the instance of a mathematical operation
  - Edge: multi-dimensional data set (tensors) on which the operations are performed.

---

# Logistic Regression

$$ z = w^Tx+b $$
$$ \hat{y} = a = \sigma(z) = \frac{1}{1+e^{-z}} $$
$$ \mathcal{L}(a,y) = -(y\log{a} + (1-y)\log{(1-a)}) $$
$$ \frac{\sigma{\mathcal{L}(a,y)}}{\sigma{a}} = -\frac{y}{a}-\frac{1-y}{1-a} $$
---

$$ \frac{\sigma{\mathcal{L}(a,y)}}{\sigma{z}} = \frac{\sigma{\mathcal{L}(a,y)}}{\sigma{a}} \cdot \frac{\sigma{a}}{\sigma{z}} $$
$$ \hat{y} = a = \sigma(z) = \frac{1}{1+e^{-z}} $$
$$ \frac{\sigma{a}}{\sigma{z}} = a(1-a) $$

--- 

### WTF??

$$ 
\frac{d}{dx}(\frac{1}{1+e^{-x}}) 
= \frac{d}{dx}((1+e^{-x})^{-1}) 
$$
$$
\frac{d}{dx}f(g(x)) = f'(g(x))g`(x) \because Chain Rule
$$
$$
f`(x) = x^{-1},g`(x) = 1-e^{-x}=u
$$
$$
\frac{d}{du}(u^{-1})\frac{d}{dx}(1-e^{-x})
$$

<!-- _footer: Solution - https://www.mathway.com/ko/popular-problems/Calculus/538330 -->
---

$$
= -\frac{1}{(1+e^{-x})^2}\frac{d}{dx}(1+e^{-x}) 
$$

$$ 
= \frac{1}{(1+e^{-x})^2}(-e^{-x})
= \frac{e^{-x}}{(1+e^{-x})^2} $$

$$
= \frac{1-1+e^{-x}}{(1+e^{-x})^2} = \frac{1+e^{-x}}{(1+e^{-x})^2}-\frac{1}{(1+e^{-x})^2}
$$
$$ 
= \frac{1}{1+e^{-x}} \cdot (1-\frac{1}{1+e^{-x}}) = \sigma(x)(1-\sigma(x))
$$

<!-- _footer: Solution - https://medium.com/analytics-vidhya/derivative-of-log-loss-function-for-logistic-regression-9b832f025c2d -->

---

$$ \frac{\sigma{\mathcal{L}(a,y)}}{\sigma{z}} = \frac{\sigma{\mathcal{L}(a,y)}}{\sigma{a}} \cdot \frac{\sigma{a}}{\sigma{z}} $$
$$
= (-\frac{y}{a}-\frac{1-y}{1-a}) \cdot a(1-a)
$$
$$
= -y(1-a)+a(1-y)
$$
$$ = -y +ay+a-ay $$
$$ = a-y $$
---

$$ z = w^Tx+b $$
$$ 
\sigma{w_i}
= \frac{\sigma{\mathcal{L}}}{\sigma{w_{i}}} 
= \frac{\sigma{\mathcal{L}}}{\sigma{z}} \cdot \frac{\sigma{\mathcal{z}}}{\sigma{w_{i}}} 
= \sigma{z} \cdot x_{i}
$$
$$ 
\sigma{b} =
\frac{\sigma{\mathcal{L}}}{\sigma{b}} 
= \frac{\sigma{\mathcal{L}}}{\sigma{z}} \cdot \frac{\sigma{\mathcal{z}}}{\sigma{b}}
= \sigma{z}
$$

---

## Logistic Regression in m examples

$$
J(w,b) = \frac{1}{m}\sum_{i=1}^{m}\mathcal{L}(a^i,y^i)
$$
$$
\to a^i=\hat{y}^i=\sigma(z^i)=\sigma(w^ix^i+b)
$$

$$ 
\frac{\sigma}{\sigma{w_i}}J(w,b)=\frac{1}{m}\sum_{i=1}^{m}\frac{\sigma}{\sigma{w_i}}\mathcal{L}(a^i,y^i)
$$

---
## Coding
```python
J,dw,db=0,0,0
for i in range(m):
    z = w*x+b
    a = d(z)
    J += -(y*log(a^i)+(1-y)*log(1-a^i))
    dz = a-y
    dw += x*dz
    db += dz
J/=m
dw/=m
db/=m
```
- [Note] Vectorization is used for the performance
---

# References

- Computational Graph: https://www.tutorialspoint.com/python_deep_learning/python_deep_learning_computational_graphs.htm
- Data Flow Graph: https://yamerong.tistory.com/68
- Intro to Graph: https://www.tensorflow.org/guide/intro_to_graphs

---

<!-- class: lead -->

# Q & A
