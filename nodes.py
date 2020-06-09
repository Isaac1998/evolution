import matplotlib.lines as lines
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import random


class Nodes:

    def __init__(self,
                 x_nodes=11,
                 y_nodes=11,
                 x_size=1,
                 y_size=1):

        step_x = x_size / (x_nodes - 1)
        step_y = y_size / (y_nodes - 1)
        cols_n = ["X", "Y", "N1", "N2", "N3", "N4", "N5", "N6"]
        self.nodes = pd.DataFrame(columns=cols_n)
        self.num_nodes = 2 * x_nodes * y_nodes - (x_nodes + y_nodes) + 1

        for i in range(self.num_nodes):
            if (i % (2 * x_nodes - 1) < x_nodes):
                newf = pd.DataFrame([[(i % (2 * x_nodes - 1)) * step_x,
                                      (i // (2 * x_nodes - 1)) * step_y,
                                      -1, -1, -1, -1, -1, -1]],
                                    columns=cols_n)
            else:
                newf = pd.DataFrame([[((i % (2 * x_nodes - 1)) % x_nodes + 0.5) * step_x,
                                      (i // (2 * x_nodes - 1) + 0.5) * step_y,
                                      -1, -1, -1, -1, -1, -1]],
                                    columns=cols_n)
            self.nodes = self.nodes.append(newf, ignore_index=True)

        self.nodes["N1"] = self.nodes.index + 1
        self.nodes["N2"] = self.nodes.index - 1
        self.nodes["N3"] = self.nodes.index + x_nodes - 1
        self.nodes["N4"] = self.nodes.index + x_nodes
        self.nodes["N5"] = self.nodes.index - x_nodes + 1
        self.nodes["N6"] = self.nodes.index - x_nodes

        self.nodes.loc[self.nodes["X"] == 0, "N2"] = -1
        self.nodes.loc[self.nodes["X"] == 0, "N3"] = self.nodes.loc[
                                                         self.nodes["X"] == 0].index + int(2 * x_nodes - 1)
        self.nodes.loc[(self.nodes["X"] == 0) | (self.nodes["Y"] == 0), "N6"] = -1

        self.nodes.loc[self.nodes["X"] == x_size, "N1"] = -1
        self.nodes.loc[self.nodes["X"] == x_size, "N4"] = self.nodes.loc[
                                                              self.nodes["X"] == x_size].index + int(2 * x_nodes - 1)
        self.nodes.loc[(self.nodes["X"] == x_size) | (self.nodes["Y"] == 0), "N5"] = -1

        self.nodes.loc[self.nodes["Y"] == y_size, "N3"] = -1
        self.nodes.loc[self.nodes["Y"] == y_size, "N4"] = -1

        self.nodes.loc[((self.nodes["X"] < (1.01 * step_x / 2))
                        & (self.nodes["X"] > (0.99 * step_x / 2))), "N2"] = -1
        self.nodes.loc[((self.nodes["X"] < (x_size - 0.99 * step_x / 2))
                        & (self.nodes["X"] > (x_size - 1.01 * step_x / 2))), "N1"] = -1
        self.nodes = self.nodes.convert_dtypes()


class Edges:

    def __init__(self, nodes):

        cols_e = ["N1", "N2", "W"]
        self.edges = pd.DataFrame(columns=cols_e)
        for i in nodes.index:
            for nod in nodes.columns[2:]:
                if (nodes.loc[i, nod] < 0): continue
                newf = pd.DataFrame([[i, nodes.loc[i, nod], 0]], columns=cols_e)
                self.edges = self.edges.append(newf, ignore_index=True)
        self.edges = self.edges.drop(self.edges[self.edges["N2"] < self.edges["N1"]].index)
        self.edges.index = range(self.edges.shape[0])


class Trians:

    def __init__(self, nodes, edges):

        cols_t = ["N1", "N2", "N3", "E1", "E2", "E3", "V", "F"]
        self.trians = pd.DataFrame(columns=cols_t)
        for ind, ser in nodes.iterrows():
            serl = ser[ser >= 0].to_list()[2:]
            serl.sort()
            for i in range(len(serl)):
                curr = nodes.loc[serl[i]]
                k = curr[curr.isin(serl[i + 1:])].dropna()
                if k.empty: break
                newf = pd.DataFrame([[ind, serl[i], k.values[0], None, None, None, 0, 0]],
                                    columns=cols_t)
                self.trians = self.trians.append(newf, ignore_index=True)

        self.trians = self.trians.drop(self.trians[self.trians["N2"] < self.trians["N1"]].index)
        self.trians.index = range(self.trians.shape[0])

        for ind, ser in edges.iterrows():
            self.trians.loc[((self.trians["N1"] == ser["N1"]) & (self.trians["N2"] == ser["N2"])), "E1"] = ind
            self.trians.loc[((self.trians["N1"] == ser["N1"]) & (self.trians["N3"] == ser["N2"])), "E2"] = ind
            self.trians.loc[((self.trians["N2"] == ser["N1"]) & (self.trians["N3"] == ser["N2"])), "E3"] = ind


class Domain:

    def __init__(self,
                 x_size=1,
                 y_size=1,
                 tritics=10):
        self.xs = x_size
        self.ys = y_size
        self.ns = Nodes(x_size=x_size, y_size=y_size)
        self.ed = Edges(nodes=self.ns.nodes)
        self.tr = Trians(nodes=self.ns.nodes,
                         edges=self.ed.edges)
        self.trics = tritics
        for i in range(self.trics):
            center = random.randint(0, self.tr.trians.shape[0])
            self.tritic_V(center)
            curr = self.tr.trians.loc[center]
            for j in self.tr.trians.loc[self.tr.trians["N1"] == curr["N1"]].index:
                self.tritic_F(j)
            for j in self.tr.trians.loc[self.tr.trians["N2"] == curr["N1"]].index:
                self.tritic_F(j)
            for j in self.tr.trians.loc[self.tr.trians["N3"] == curr["N1"]].index:
                self.tritic_F(j)
            for j in self.tr.trians.loc[self.tr.trians["N1"] == curr["N2"]].index:
                self.tritic_F(j)
            for j in self.tr.trians.loc[self.tr.trians["N2"] == curr["N2"]].index:
                self.tritic_F(j)
            for j in self.tr.trians.loc[self.tr.trians["N3"] == curr["N2"]].index:
                self.tritic_F(j)
            for j in self.tr.trians.loc[self.tr.trians["N1"] == curr["N3"]].index:
                self.tritic_F(j)
            for j in self.tr.trians.loc[self.tr.trians["N2"] == curr["N3"]].index:
                self.tritic_F(j)
            for j in self.tr.trians.loc[self.tr.trians["N3"] == curr["N3"]].index:
                self.tritic_F(j)

    def tritic_V(self, number):
        curr = self.tr.trians.loc[number]
        self.tr.trians.loc[number, "V"] = 1
        self.ed.edges.loc[curr["E1"], "W"] = 1
        self.ed.edges.loc[curr["E2"], "W"] = 1
        self.ed.edges.loc[curr["E3"], "W"] = 1

    def tritic_F(self, number):
        curr = self.tr.trians.loc[number]
        self.tr.trians.loc[number, "F"] = 1
        self.ed.edges.loc[curr["E1"], "W"] = 1
        self.ed.edges.loc[curr["E2"], "W"] = 1
        self.ed.edges.loc[curr["E3"], "W"] = 1

    def random_steps(self):
        for ind, ser in self.ns.nodes.iterrows():
            x0 = ser["X"]
            y0 = ser["Y"]
            if ((x0 == 0) | (y0 == 0) | (x0 == self.xs) | (y0 == self.ys)): continue
            rmin = 2
            for neigh in ser["N1":"N6"]:
                if (neigh < 0): continue
                x1 = self.ns.nodes.loc[neigh, "X"]
                y1 = self.ns.nodes.loc[neigh, "Y"]
                r = np.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0))
                if (r < rmin): rmin = r
            cur = "N" + str(random.randint(1, 6))
            if (ser[cur] < 0): continue
            x1 = self.ns.nodes.loc[ser[cur], "X"]
            y1 = self.ns.nodes.loc[ser[cur], "Y"]
            r = np.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0))
            delta = rmin * (random.random() / 10)
            self.ns.nodes.loc[ind, "X"] = x0 + (delta / r) * (x1 - x0)
            self.ns.nodes.loc[ind, "Y"] = y0 + (delta / r) * (y1 - y0)


dom = Domain()
for i in range(10):
    dom.random_steps()
nodes = dom.ns.nodes
edges = dom.ed.edges
trians = dom.tr.trians

fig, ax = plt.subplots()

for ind, ser in trians.iterrows():
    tri = patches.Polygon([[nodes.loc[ser["N1"], "X"],
                            nodes.loc[ser["N1"], "Y"]],
                           [nodes.loc[ser["N2"], "X"],
                            nodes.loc[ser["N2"], "Y"]],
                           [nodes.loc[ser["N3"], "X"],
                            nodes.loc[ser["N3"], "Y"]]],
                          color=[ser["V"], 0, ser["F"]])
    ax.add_artist(tri)

for ind, ser in edges.iterrows():
    line = lines.Line2D([nodes.loc[ser["N1"], "X"],
                         nodes.loc[ser["N2"], "X"]],
                        [nodes.loc[ser["N1"], "Y"],
                         nodes.loc[ser["N2"], "Y"]],
                        lw=0.5, color=str(ser["W"]), axes=ax)
    ax.add_line(line)
ax.axis('square')
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
plt.savefig("time"+str(0)+".png", dpi=1200,bbox_inches="tight",figsize=[10,10])
plt.show()
