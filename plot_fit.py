def plot_result(x, y, res, mask1=None, mask2=None):

    fig, ax = plt.subplots()

    ax.plot(x, y, label="data", linewidth=2)

    # ---- optional region highlight ----
    if mask1 is not None:
        ax.plot(x[mask1], y[mask2], color="orange", alpha=0.6)

    if mask2 is not None:
        ax.plot(x[mask1], y[mask2], color="purple", alpha=0.6)

    # ---- initial guesses ----
    ax.plot(x, res["peak1_guess"], "--", color="gray", alpha=0.5,)
    ax.plot(x, res["peak2_guess"], "--", color="gray", alpha=0.5,)

    # ---- fitted peaks ----
    ax.plot(x, res["peak1_fit"], color="orange", label="LDPE fit")
    ax.plot(x, res["peak2_fit"], color="purple", label="HDPE fit")
    ax.plot(x, res["total_fit"], color="red", label="total fit")

    # ---- filled areas under peaks ----
    ax.fill_between(x, res["peak1_fit"], color="orange", alpha=0.3)
    ax.fill_between(x, res["peak2_fit"], color="purple", alpha=0.3)

    # ---- residual ----
    ax.plot(x, y - res["total_fit"] - 0.1, color="black", label="residual")
    ax.plot(x, np.zeros_like(x) - 0.1, color="gray", alpha=0.5)

    ax.legend()
    plt.show()
