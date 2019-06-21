## Optimizers
--------------

Optimizers minimize an objective function (`op::Operation`) with respect to several variables (`op::Operation`). All optimizers inherit from `optimizer::Optimizer<T>` class and define a `minimize` function which takes a vector of variables to minimize w.r.t. 

For example, consider the following code, which minimizes the expression `x^2 + c` w.r.t. `x` using _Stochastic Gradient Descent_.

```c++
/* initialize our expression */
auto x = op::var<double> ("x", {1}, {CONSTANT, {5.0}}, mem);
auto c = op::var<double> ("c", {1}, {CONSTANT, {10.0}}, mem);
auto expr = op::add(op::pow(x,2), c);

/* create our optimizer */
double learning_rate = 0.05;
optimizer::GradientDescent<double> optim (expr, learning_rate);

unsigned int n_iter = 100;
for (unsigned int i = 0; i < n_iter; i++) {
    optim.minimize({x});
}
```

This should bring the value of `x->eval()` close to `0` and `expr->eval()` to `10`. The accuracy of this is dependent on the learning rate and number of iterations. For a neural network a learning rate around `0.01-0.1` is typically sufficient and the number of training iterations is very model dependent.