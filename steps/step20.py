import numpy as np
import weakref
import contextlib

class Config:
    enable_backdrop = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name) # 今のConfigのnameつまりenable_backdropをgetする．
    setattr(Config, name, value) # enable_backdropをvalueにsetする．
    try:
        yield # 一旦関数の処理を止める．つまりwithで呼び出したそれ以降を実行する．

    finally: # withの処理が終わったら元の値（True)にsetする
        setattr(Config, name, old_value)

# 毎回return部分を書くのが面倒なので作った関数
def no_grad():
    return using_config('enable_backdrop', False)


class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property # メソッドを属性のようにアクセスできるようになる．
    def shape(self):
        return self.data.shape
    
    @property 
    def ndim(self):
        return self.data.ndim
    
    @property 
    def size(self):
        return self.data.size
    
    @property 
    def dtype(self):
        return self.data.dtype
    
    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    
    def backward(self, retain_grad = False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)
        while funcs:
            f = funcs.pop()
            # gys = [output.grad for output in f.outputs]
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx           
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:                    
                    add_func(x.creator)
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def cleargrad(self):
        self.grad = None

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        if Config.enable_backdrop:
            
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()
    
    def backward(self, gys):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):        
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx
    
class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0
    
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def add(x0, x1):
    return Add()(x0, x1)

def mul(x0, x1):
    return Mul()(x0, x1)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

# with no_grad():
#     x = Variable(np.array(2.0))
#     y = square(x)
#     print(y.data)

Variable.__mul__ = mul
Variable.__add__ = add

a = Variable(np.array(3.0))
b = Variable(np.array(2.0))
c = Variable(np.array(1.0))

# y = add(mul(a, b), c)
y = a * b + c
y.backward()

print(y)
print(a.grad)
print(b.grad)