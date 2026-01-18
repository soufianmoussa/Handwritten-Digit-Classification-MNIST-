import numpy as np

def sigmoid(z):
    """
    Computes the sigmoid of z.
    """
    # Clip z to avoid overflow
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def load_weights(path='model_weights.npz'):
    """
    Loads the trained weights from a .npz file.
    Expects 'Theta1' and 'Theta2' keys.
    """
    try:
        data = np.load(path)
        return data['Theta1'], data['Theta2']
    except FileNotFoundError:
        return None, None

def predict_with_confidence(X, Theta1, Theta2):
    """
    Predict the label and confidence of an input.
    """
    m = X.shape[0]
    
    # Layer 1
    a1 = np.concatenate([np.ones((m, 1)), X], axis=1)
    z2 = np.dot(a1, Theta1.T)
    a2 = sigmoid(z2)
    
    # Layer 2
    a2 = np.concatenate([np.ones((m, 1)), a2], axis=1)
    z3 = np.dot(a2, Theta2.T)
    h = sigmoid(z3)
    
    p = np.argmax(h, axis=1)
    prob = np.max(h, axis=1)
    
    return p, prob

def predict(X, Theta1, Theta2):
    return predict_with_confidence(X, Theta1, Theta2)
