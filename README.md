# Implicit Deep Learning

This is an implementation of the implicit deep learning presented in the following paper.

```
@misc{ghaoui2020implicit,
      title={Implicit Deep Learning}, 
      author={Laurent El Ghaoui and Fangda Gu and Bertrand Travacca and Armin Askari and Alicia Y. Tsai},
      year={2020},
      eprint={1908.06315},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

Runnable scripts to train an implicit model include

* `cifar_implicit_grad_inf.py`
* `mnist_implicit_grad_inf.py`

Visualization of a ResNet as an implicit model can be found in `implicit/ResNetImplicitConstructon.ipynb`.

The code was tested under pytorch version 1.4.0.

