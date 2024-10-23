import numpy as np
from sklearn.datasets import make_swiss_roll,make_s_curve
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
if __name__ == "__main__":
    noise = 0.05
    n_samples = 5_000
    X, t = make_swiss_roll(noise=noise, random_state=0, n_samples=n_samples)
    # # X, t = make_s_curve(noise=noise,random_state=0,n_samples=n_samples)
    # # X = X * 5
    # # fig = plt.figure()
    # # ax = fig.add_subplot(111, projection="3d", elev=3, azim=-70)
    # # ax.scatter(X[:, 0], X[:, 1], X[:, 2],c='g')
    # # fig.savefig("data_3d.png")
    # fig = px.scatter_3d(X[:, 0], X[:, 1], X[:, 2])
    # # fig.show()
    # fig.write_html("f.html")

    import plotly.graph_objects as go
    #
    # fig = go.Figure(
    #     data=go.Scatter3d(X[:, 0], X[:, 1], X[:, 2]),
    #     layout_title_text="A Figure Displayed with fig.show()"
    # )
    # fig.show()

    t = np.linspace(0, 10, 50)
    x, y, z = np.cos(t), np.sin(t), t

    fig = go.Figure(data=[go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2],
                                       mode='markers')])
    # fig.show()
    fig.write_html("f.html")
