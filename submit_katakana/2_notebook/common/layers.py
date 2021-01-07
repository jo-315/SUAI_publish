import numpy as np
from collections import OrderedDict

def cross_entropy_error(y, t):
    """
    y : 出力値(通常は、0-1の確率)  
    t : 正解値(通常は、0or1)  
    """
    if y.ndim==1:
        t = t.reshape(1, -1)
        y = y.reshape(1, -1)
        
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum( t * np.log(y + delta)) / batch_size

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))

class ReLU:
    def __init__(self):
        self.mask = None
        
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy() #参照渡しではなく複製する
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        """
        dout : float, 上流(出力)側の勾配
        """        
        dout[self.mask] = 0
        dLdx = dout
        return dLdx
    
class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b
        
        # 初期値
        self.x = None
        self.original_x_shape = None
        self.dW = None # 重みの微分
        self.db = None # バイアスの微分

    def forward(self, x):
        """
        順伝播
        """
        # 値の保持
        self.x = x

        # 順伝播
        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        """
        逆伝播
        dout : float, 上流(出力)側の勾配
        """
        # dxは前の層に伝える必要がある
        dx = np.dot(dout, self.W.T)
        
        # dWとdbは、勾配法の計算に使われるので、値を保持しておく
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0) # N個の合計になる
        
        return dx
    
class SoftmaxWithLoss:
    def __init__(self):
        
        # 初期値
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ

    def forward(self, x, t):
        """
        順伝播
        """
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        """
        逆伝播
        伝播する値をバッチサイズで割ること
        """
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx

    
    
def numerical_gradient(f, W):
    """
    全ての次元について、個々の次元だけの微分を求める
    f : 関数
    W : 偏微分を求めたい場所の座標。多次元配列
    """
    h = 1e-4 # 0.0001
    grad = np.zeros_like(W)
    
    it = np.nditer(W, flags=['multi_index'])
    
    while not it.finished:
        idx = it.multi_index
        tmp_val = W[idx]
        
        W[idx] = tmp_val + h
        fxh1 = f(W)
        
        W[idx] = tmp_val - h 
        fxh2 = f(W)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        W[idx] = tmp_val # 値を元に戻す
        
        # 次のindexへ進める
        it.iternext()   
        
    return grad

class BatchNormalization:
    def __init__(self, gamma, beta, rho=0.9, moving_mean=None, moving_var=None):
        self.gamma = gamma # スケールさせるためのパラメータ, 学習によって更新させる.
        self.beta = beta # シフトさせるためのパラメータ, 学習によって更新させる
        self.rho = rho # 移動平均を算出する際に使用する係数

        # 予測時に使用する平均と分散
        self.moving_mean = moving_mean # muの移動平均
        self.moving_var = moving_var        # varの移動平均
        
        # 計算中に算出される値を保持しておく変数群
        self.batch_size = None
        self.x_mu = None
        self.x_std = None        
        self.std = None
        self.dgamma = None
        self.dbeta = None
        
    def forward(self, x, train_flg=True, epsilon=1e-8):
        """
        x : 入力. n×dの行列. nはあるミニバッチのバッチサイズ. dは手前の層のノード数
        """
        if (self.moving_mean is None) or (self.moving_var is None):
            N, D = x.shape
            self.moving_mean = np.zeros(D)
            self.moving_var = np.zeros(D)
       
        if train_flg:
            """
            学習時
            """
            # 入力xについて、nの方向に平均値を算出. 
            mu = np.mean(x, axis=0) # 要素数d個のベクトル
            
            # 入力xから平均値を引く
            x_mu = x - mu   # n*d行列
            
            # 入力xの分散を求める
            var = np.mean(x_mu**2, axis=0)  # 要素数d個のベクトル
            
            # 入力xの標準偏差を求める(epsilonを足してから標準偏差を求める)
            std = np.sqrt(var + epsilon)  # 要素数d個のベクトル
            
            # 標準化
            x_std = x_mu / std  # n*d行列
            
            # 値を保持しておく
            self.batch_size = x.shape[0]
            self.x_mu = x_mu
            self.x_std = x_std
            self.std = std
            self.moving_mean = self.rho * self.moving_mean + (1-self.rho) * mu
            self.moving_var = self.rho * self.moving_var + (1-self.rho) * var
        else:
            """
            予測時
            """
            x_mu = x - self.moving_mean # n*d行列
            x_std = x_mu / np.sqrt(self.moving_var + epsilon) # n*d行列
            
        # gammaでスケールし、betaでシフトさせる
        out = self.gamma * x_std + self.beta # n*d行列
        return out

    def backward(self, dout):
        # betaの勾配
        dbeta = np.sum(dout, axis=0)
        
        # gammaの勾配(n方向に合計)
        dgamma = np.sum(self.x_std * dout, axis=0)
        
        # Xstdの勾配
        a1 = self.gamma * dout
        
        # Xmuの勾配(1つ目)
        a2 = a1 / self.std
        
        # 標準偏差の逆数の勾配(n方向に合計)
        a3 = np.sum(a1 * self.x_mu, axis=0)

        # 標準偏差の勾配
        a4 = -(a3) / (self.std * self.std)
        
        # 分散の勾配
        a5 = 0.5 * a4 / self.std
        
        # Xmuの2乗の勾配
        a6 = a5 / self.batch_size
        
        # Xmuの勾配(2つ目)
        a7 = 2.0  * self.x_mu * a6
        
        # muの勾配
        a8 = np.sum(-(a2+a7), axis=0)

        # Xの勾配
        dx = a2 + a7 +  a8 / self.batch_size # 第3項はn方向に平均
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx
    
class Dropout:
    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask