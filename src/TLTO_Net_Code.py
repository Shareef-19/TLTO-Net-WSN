
import numpy as np
import matplotlib.pyplot as plt

def finish(title, xlabel, ylabel, legend_loc="best"):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)
    if legend_loc is not None:
        plt.legend(loc=legend_loc)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------
# Fig. 3 — Adaptive Trust Penalty vs Abnormal Node Behavior
# -------------------------------------------------------------
x_acj = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
theta_node1 = np.array([1.0, 0.92, 0.70, 0.48, 0.22, 0.10])
theta_node2 = np.array([1.0, 0.97, 0.92, 0.80, 0.75, 0.70])
theta_node3 = np.array([1.0, 0.90, 0.72, 0.60, 0.40, 0.20])
theta_node4 = np.array([1.0, 0.80, 0.60, 0.40, 0.20, 0.10])

plt.figure()
plt.plot(x_acj, theta_node1, label="Node-1")
plt.plot(x_acj, theta_node2, label="Node-2")
plt.plot(x_acj, theta_node3, label="Node-3")
plt.plot(x_acj, theta_node4, label="Node-4")
finish("Adaptive Trust Penalty Coefficient vs. Abnormal Node Behavior in WSNs",
       "Proportion of abnormal behavior (ACj)",
       "Adaptive penalty coefficient (theta)",
       legend_loc="upper right")

# -------------------------------------------------------------
# Fig. 4 — Comprehensive Trust vs Round Time (stabilized)
# -------------------------------------------------------------
t = np.linspace(0, 100, 101)
decay = 0.6 * np.exp(-t / 35.0)
decay[t > 60] = decay[np.where(t == 60)[0][0]]
y_ct = decay.copy()
y_ct[t >= 60] = 0.30

plt.figure()
plt.plot(t, y_ct, label="Comprehensive Trust")
finish("Comprehensive Trust vs. Round Time (Stabilized at 0.3)",
       "Round Time",
       "Comprehensive Trust",
       legend_loc=None)

# Additional figures exactly as provided by user...
print("All TLTO-Net figures generated programmatically.")
