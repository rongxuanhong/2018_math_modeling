######################### 最大生成树算法 ######################
class Kruskal(object):
    def __init__(self, graph):
        self.graph = graph
        self.parent = dict()
        self.rank = dict()

    def make_set(self, vertice):
        self.parent[vertice] = vertice
        self.rank[vertice] = 0

    def find(self, vertice):
        """
        递归找到顶点的父节点或者祖先结点
        :param vertice:
        :return:
        """
        if self.parent[vertice] != vertice:
            self.parent[vertice] = self.find(self.parent[vertice])  # 递归找到vertice的祖先结点
        return self.parent[vertice]

    def union(self, root1, root2):
        """
        合并
        :param root1:
        :param root2:
        :return:
        """
        if self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1  # 排名高的作为父节点
        else:
            self.parent[root1] = root2
            if self.rank[root1] == self.rank[root2]: self.rank[root2] += 1  # 相等则提高父节点的排名

    def kruskal(self):
        """
        算法主要步骤
        :return:
        """
        for vertice in self.graph['vertices']:
            self.make_set(vertice)

        maximum_spanning_tree = set()
        edges = list(set(self.graph['edges']))
        edges.sort(reverse=True)  # 降序排列边
        for edge in edges:
            weight, vertice1, vertice2 = edge
            root1, root2 = self.find(vertice1), self.find(vertice2)  # 找到顶点的祖先结点
            if root1 != root2:  # 不构成环
                self.union(root1, root2)  # 合并边
                maximum_spanning_tree.add(edge)
        return maximum_spanning_tree
