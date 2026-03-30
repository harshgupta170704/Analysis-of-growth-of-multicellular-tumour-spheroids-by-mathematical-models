# ============================================
# PINN for Tumor Growth (TensorFlow Version)
# ============================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ============================================
# 1. DATASET
# ============================================

data = np.array([
    [3.46,0.0158],[4.58,0.0264],[5.67,0.0326],[6.64,0.0445],
    [7.63,0.0646],[8.41,0.0933],[9.32,0.1454],[10.27,0.2183],
    [11.19,0.2842],[12.39,0.4977],[13.42,0.6033],[15.19,0.8441],
    [16.24,1.2163],[17.23,1.4470],[18.18,2.3298],[19.29,2.5342],
    [21.23,3.0064],[21.99,3.4044],[24.33,3.2046],[25.58,4.5241],
    [26.43,4.3459],[27.44,5.1374],[28.43,5.5376],[30.49,4.8946],
    [31.34,5.0660],[32.34,6.1494],[33.00,6.8548],[35.20,5.9668],
    [36.34,6.6945],[37.29,6.6395],[38.50,6.8971],[39.67,7.2966],
    [41.37,7.2268],[42.58,6.8815],[45.39,8.0993],[46.38,7.2112],
    [48.29,7.0694],[49.24,7.4971],[50.19,6.9974],[51.14,6.7219],
    [52.10,7.0523],[54.00,7.1095],[56.33,7.0694],[57.33,8.0562],
    [59.38,7.2268]
])

t = tf.convert_to_tensor(data[:,0].reshape(-1,1), dtype=tf.float32)
y = tf.convert_to_tensor(data[:,1].reshape(-1,1), dtype=tf.float32)

# ============================================
# 2. MODEL (Neural Network)
# ============================================

class PINN(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.hidden1 = tf.keras.layers.Dense(64, activation='tanh')
        self.hidden2 = tf.keras.layers.Dense(64, activation='tanh')
        self.out = tf.keras.layers.Dense(1)

        # Trainable parameters
        self.k = tf.Variable(0.5, dtype=tf.float32)
        self.C = tf.Variable(7.0, dtype=tf.float32)
        self.theta = tf.Variable(0.5, dtype=tf.float32)

    def call(self, t):
        x = self.hidden1(t)
        x = self.hidden2(x)
        return self.out(x)

# ============================================
# 3. RESIDUAL FUNCTIONS
# ============================================

def residual_verhulst(model, t):
    with tf.GradientTape() as tape:
        tape.watch(t)
        p = model(t)
    dp_dt = tape.gradient(p, t)

    return dp_dt - model.k * p * (1 - p/model.C)


def residual_montroll(model, t):
    with tf.GradientTape() as tape:
        tape.watch(t)
        p = model(t)
    dp_dt = tape.gradient(p, t)

    return dp_dt - model.k * p * (1 - (p/model.C)**model.theta)

# ============================================
# 4. TRAINING FUNCTION
# ============================================

def train(model, residual_fn, epochs=5000):

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_history = []

    for epoch in range(epochs):

        with tf.GradientTape() as tape:

            # Data loss
            pred = model(t)
            loss_data = tf.reduce_mean((pred - y)**2)

            # Physics loss
            res = residual_fn(model, t)
            loss_physics = tf.reduce_mean(res**2)

            # Total loss
            loss = loss_data + loss_physics

        grads = tape.gradient(loss, model.trainable_variables + [model.k, model.C, model.theta])
        optimizer.apply_gradients(zip(grads, model.trainable_variables + [model.k, model.C, model.theta]))

        loss_history.append(loss.numpy())

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy()}")

    return loss_history

# ============================================
# 5. TRAIN VERHULST MODEL
# ============================================

print("\nTraining Verhulst Model...")
model_v = PINN()
loss_v = train(model_v, residual_verhulst)

# ============================================
# 6. TRAIN MONTROLL MODEL
# ============================================

print("\nTraining Montroll Model...")
model_m = PINN()
loss_m = train(model_m, residual_montroll)

# ============================================
# 7. VISUALIZATION
# ============================================

t_test = np.linspace(min(data[:,0]), max(data[:,0]), 200).reshape(-1,1)

y_v = model_v(tf.convert_to_tensor(t_test, dtype=tf.float32))
y_m = model_m(tf.convert_to_tensor(t_test, dtype=tf.float32))

plt.figure()
plt.scatter(data[:,0], data[:,1], label="Data")
plt.plot(t_test, y_v.numpy(), label="Verhulst PINN")
plt.plot(t_test, y_m.numpy(), label="Montroll PINN")
plt.legend()
plt.title("TensorFlow PINN Model Comparison")
plt.xlabel("Time")
plt.ylabel("Tumor Volume")
plt.show()

# ============================================
# 8. PARAMETERS
# ============================================

print("\nVerhulst Parameters:")
print("k =", model_v.k.numpy())
print("C =", model_v.C.numpy())

print("\nMontroll Parameters:")
print("k =", model_m.k.numpy())
print("C =", model_m.C.numpy())
print("theta =", model_m.theta.numpy())
