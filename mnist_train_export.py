#!/usr/bin/env python3
"""
Download MNIST, train a tiny 2-layer MLP (784 -> 64 -> 10) with pure
numpy SGD, export the weights as a C header for the bare-metal board.

Also saves the first 20 test images as PNG so we can stream them back
for recognition.
"""
import gzip, struct, urllib.request, os, sys, time
import numpy as np

OUT_DIR = '/tmp/fz3a_dp'
CACHE   = os.path.join(OUT_DIR, 'mnist_cache')
H_FILE  = os.path.join(OUT_DIR, 'mnist_weights.h')
PNG_DIR = os.path.join(OUT_DIR, 'digit_pngs')
os.makedirs(CACHE, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)

MIRROR = 'https://storage.googleapis.com/cvdf-datasets/mnist'
FILES = {
    'train_images': 'train-images-idx3-ubyte.gz',
    'train_labels': 'train-labels-idx1-ubyte.gz',
    'test_images':  't10k-images-idx3-ubyte.gz',
    'test_labels':  't10k-labels-idx1-ubyte.gz',
}

def download(name, fname):
    path = os.path.join(CACHE, fname)
    if not os.path.exists(path):
        url = f'{MIRROR}/{fname}'
        print(f'[dl  ] {url}')
        urllib.request.urlretrieve(url, path)
    return path

def read_idx(path):
    with gzip.open(path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        if magic == 2051:   # images
            n, h, w = struct.unpack('>III', f.read(12))
            data = np.frombuffer(f.read(n * h * w), dtype=np.uint8)
            return data.reshape(n, h * w)
        elif magic == 2049: # labels
            n = struct.unpack('>I', f.read(4))[0]
            return np.frombuffer(f.read(n), dtype=np.uint8)
        else:
            raise ValueError(f'bad magic {magic}')

def load_mnist():
    paths = {k: download(k, v) for k, v in FILES.items()}
    X_tr = read_idx(paths['train_images']).astype(np.float32) / 255.0
    y_tr = read_idx(paths['train_labels']).astype(np.int64)
    X_te = read_idx(paths['test_images']).astype(np.float32) / 255.0
    y_te = read_idx(paths['test_labels']).astype(np.int64)
    return X_tr, y_tr, X_te, y_te

# ---------- network (784 -> 64 -> 10 ReLU) ----------
rng = np.random.default_rng(42)
HIDDEN = 64

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def main():
    print('[mnist] loading...')
    X_tr, y_tr, X_te, y_te = load_mnist()
    print(f'[mnist] train={X_tr.shape} test={X_te.shape}')

    npz_path = os.path.join(OUT_DIR, 'mnist_weights.npz')
    if os.path.exists(npz_path):
        print(f'[load] reusing {npz_path}')
        npz = np.load(npz_path)
        W1, b1, W2, b2 = npz['W1'], npz['b1'], npz['W2'], npz['b2']
        z1 = X_te @ W1 + b1; h1 = np.maximum(z1, 0); z2 = h1 @ W2 + b2
        acc = (z2.argmax(axis=1) == y_te).mean()
        print(f'[load] cached weights acc = {acc*100:.2f}%')
        export_weights(W1, b1, W2, b2, acc)
        save_test_pngs(X_te, y_te)
        return

    # He-normal init
    W1 = rng.standard_normal((784, HIDDEN)).astype(np.float32) * np.sqrt(2.0 / 784)
    b1 = np.zeros(HIDDEN, dtype=np.float32)
    W2 = rng.standard_normal((HIDDEN, 10)).astype(np.float32) * np.sqrt(2.0 / HIDDEN)
    b2 = np.zeros(10, dtype=np.float32)

    EPOCHS = 8
    LR     = 0.1
    BS     = 128
    N = X_tr.shape[0]

    Y_tr = np.eye(10, dtype=np.float32)[y_tr]

    t0 = time.time()
    for epoch in range(EPOCHS):
        perm = rng.permutation(N)
        Xs = X_tr[perm]; Ys = Y_tr[perm]
        acc_loss = 0.0
        for i in range(0, N, BS):
            x = Xs[i:i+BS]
            y = Ys[i:i+BS]
            # forward
            z1 = x @ W1 + b1
            h1 = np.maximum(z1, 0)
            z2 = h1 @ W2 + b2
            p  = softmax(z2)
            # loss (avg cross entropy)
            loss = -np.log(p[np.arange(len(y)), y.argmax(axis=1)] + 1e-9).mean()
            acc_loss += loss * len(x)
            # backward
            dz2 = (p - y) / len(x)
            dW2 = h1.T @ dz2
            db2 = dz2.sum(axis=0)
            dh1 = dz2 @ W2.T
            dz1 = dh1 * (z1 > 0)
            dW1 = x.T @ dz1
            db1 = dz1.sum(axis=0)
            # SGD
            W2 -= LR * dW2; b2 -= LR * db2
            W1 -= LR * dW1; b1 -= LR * db1
        # eval
        z1 = X_te @ W1 + b1
        h1 = np.maximum(z1, 0)
        z2 = h1 @ W2 + b2
        acc = (z2.argmax(axis=1) == y_te).mean()
        print(f'[ep {epoch+1}/{EPOCHS}] loss={acc_loss/N:.4f}  test_acc={acc*100:.2f}%  ({time.time()-t0:.1f}s)')

    # cache for next invocation
    np.savez(npz_path, W1=W1, b1=b1, W2=W2, b2=b2)
    print(f'[save] {npz_path}')

    export_weights(W1, b1, W2, b2, acc)
    save_test_pngs(X_te, y_te)

def export_weights(W1, b1, W2, b2, acc):
    def arr_c(name, a, ctype='float'):
        s = f'static const {ctype} {name}[{a.size}] = {{\n'
        flat = a.flatten()
        for i in range(0, flat.size, 8):
            chunk = ', '.join(f'{v:+.6f}f' for v in flat[i:i+8])
            s += '    ' + chunk + (',' if i + 8 < flat.size else '') + '\n'
        s += '};\n'
        return s
    # Store W1 as [HIDDEN][784] row-major (matches board-side indexing).
    W1_hwrow = W1.T  # (HIDDEN, 784)
    # Store W2 as [10][HIDDEN] row-major.
    W2_hwrow = W2.T  # (10, HIDDEN)
    with open(H_FILE, 'w') as f:
        f.write('/* auto-generated by mnist_train_export.py - do not edit */\n')
        f.write('#ifndef MNIST_WEIGHTS_H\n#define MNIST_WEIGHTS_H\n\n')
        f.write(f'#define MNIST_HIDDEN {HIDDEN}\n\n')
        f.write(arr_c('mnist_W1', W1_hwrow))
        f.write(arr_c('mnist_b1', b1))
        f.write(arr_c('mnist_W2', W2_hwrow))
        f.write(arr_c('mnist_b2', b2))
        f.write(f'\n/* test-set accuracy at export: {acc*100:.2f}% */\n')
        f.write('\n#endif\n')
    print(f'[out ] wrote {H_FILE} ({os.path.getsize(H_FILE)/1024:.1f} KB)')

def save_test_pngs(X_te, y_te):
    try:
        from PIL import Image
        for i in range(20):
            Image.fromarray((X_te[i].reshape(28, 28) * 255).astype(np.uint8), 'L').save(
                os.path.join(PNG_DIR, f'digit_{y_te[i]}_{i}.png'))
        print(f'[out ] 20 test PNGs saved to {PNG_DIR}')
    except ImportError:
        print('[warn] PIL not available, skipping PNG export')

if __name__ == '__main__':
    main()
