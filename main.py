import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import time


class VPs:
    def __init__(self, light, *polygons, err=1e-5):
        # 排除共线情况
        self.err = err
        self.light = light + self.err * np.array([
            np.cos(r := self.err * np.random.randint(1/self.err)),
            np.sin(r)
        ])
        self.polygons = list(polygons)
        self.front_arcs = []
        self.Lighting_arcs = []
        self.Lighting_points = []
        for i, ele in enumerate(self.polygons):  # 保证都为顺时针
            if self.get_pol_area([0, 0], ele) < 0:
                self.polygons[i] = self.polygons[i][::-1]
            # 处理曲线情况（面积是0）
            pass

        self.set_arcs()
        self.set_angle()
        self.split_arcs()
        self.front_arcs = sorted(self.front_arcs, key=lambda t: t[0][2])
        self.solve()
        self.front_arcs = []
        self.Lighting_points = self.extract_list(self.Lighting_arcs)
        Lighting_points = []
        for ele in self.Lighting_points:
            if ele[3]:
                Lighting_points.append(ele)
        self.Lighting_points = Lighting_points

    def get_pol_area(self, center, polygon):
        return sum([self.to_left(center, q, b) for q, b in zip(polygon, polygon[-1:] + polygon[:-1])]) / 2

    def is_equ(self, p, q):
        return np.dot(t := p-q, t) < self.err**2

    @staticmethod
    def to_left(p, q, b):
        return np.linalg.det(np.array([
            [p[0], p[1], 1],
            [q[0], q[1], 1],
            [b[0], b[1], 1],
        ]))

    @staticmethod
    def bi_search(point, arc):
        low, high = 0, len(arc) - 1
        mid = (low + high) // 2
        while low != mid:
            if point[2] >= arc[mid][2]:
                low = mid
            else:
                high = mid
            mid = (low + high) // 2

        return low

    @staticmethod
    def extract_list(lst):
        def extract_(lst_):
            ans = []
            for ele in lst_:
                if ele.__class__ == list:
                    ans = [*ans, *extract_(ele)]
                else:
                    ans.append(ele)
            return ans
        return np.array(extract_(lst)).reshape((-1, 4))

    @staticmethod
    def rotate(vec, theta):
        return np.matmul(
            np.array([
                [+np.cos(theta), -np.sin(theta)],
                [+np.sin(theta), +np.cos(theta)],
            ]),
            np.expand_dims(vec, axis=1),
        ).squeeze(axis=1)  # .tolist()

    @staticmethod
    def shift(vec, di, dis):
        pass

    @staticmethod
    def show_pols(pols):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))
        for ele in pols:
            x, y = list(zip(*ele))
            ax.plot(x, y, 'o', color='b', markersize=5)
            ax.fill(x, y, 'r', alpha=0.3)
        plt.show()

    @staticmethod
    def norm_2(a):  # 模的平方
        return np.dot(a, a)

    @staticmethod
    def get_intersection_point(point, vec_s, vec_e):
        """
        L1: (0, 0), (point[0], point[1]) -> (x1, y1), (x2, y2)
        L2: vec_s, vec_e -> (x3, y3), (x4, y4)
        a = \begin{vmatrix} x_1 & 1 \\ x_2 & 1 \end{vmatrix}
        b = \begin{vmatrix} x_3 & 1 \\ x_4 & 1 \end{vmatrix}
        c = \begin{vmatrix} y_1 & 1 \\ y_2 & 1 \end{vmatrix}
        d = \begin{vmatrix} y_3 & 1 \\ y_4 & 1 \end{vmatrix}
        e = \begin{vmatrix} x_1 & y_1 \\ x_2 & y_2 \end{vmatrix}
        f = \begin{vmatrix} x_3 & y_3 \\ x_4 & y_4 \end{vmatrix}
        (P_x, P_y) = (\frac
        {\begin{vmatrix} e & a \\ f & b \end{vmatrix}}
        {\begin{vmatrix} a & c \\ b & d \end{vmatrix}}, \frac
        {\begin{vmatrix} e & c \\ f & d \end{vmatrix}}
        {\begin{vmatrix} a & c \\ b & d \end{vmatrix}})
        """
        x1, y1 = 0, 0
        x2, y2 = point[:2]
        x3, y3 = vec_s[:2]
        x4, y4 = vec_e[:2]
        a = np.linalg.det(np.array([[x1, 1], [x2, 1]]))
        b = np.linalg.det(np.array([[x3, 1], [x4, 1]]))
        c = np.linalg.det(np.array([[y1, 1], [y2, 1]]))
        d = np.linalg.det(np.array([[y3, 1], [y4, 1]]))
        e = np.linalg.det(np.array([[x1, y1], [x2, y2]]))
        f = np.linalg.det(np.array([[x3, y3], [x4, y4]]))
        q = np.linalg.det(np.array([[a, c], [b, d]]))
        P_x = np.linalg.det(np.array([[e, a], [f, b]]))/q
        P_y = np.linalg.det(np.array([[e, c], [f, d]]))/q
        return np.array([P_x, P_y])

    def is_intersect(self, point, vec_s, vec_e):
        return np.linalg.det(np.array([point, vec_s])) * \
               np.linalg.det(np.array([point, vec_e])) <= 0 \
               and self.to_left(vec_s, vec_e, point) < 0

    def set_arcs(self):  # 终于。。。应该没bug了。。。
        def get_front_arcs(polygon):
            edge_sign = [
                np.sign(self.to_left(self.light, q, b))
                for q, b in zip(polygon, polygon[1:] + polygon[:1])
            ]
            # print(edge_sign)

            # 第一步，找第一个极小值点，然后循环右移多边形（（（
            min_p = 0  # 极小值点
            while True:
                if edge_sign[- (min_p + 1) % len(edge_sign)] <= 0 <= \
                        edge_sign[-min_p % len(edge_sign)]:  # 找到弧头
                    break
                min_p += 1

            polygon = polygon[min_p:] + polygon[:min_p]
            edge_sign = edge_sign[min_p:] + edge_sign[:min_p]
            # print(edge_sign)

            arcs = []
            j = 0
            arc = []  # 添加起点
            while j < len(edge_sign):
                if edge_sign[j] >= 0:  # 找到弧尾
                    arc.append(polygon[j])
                elif arc:
                    arc.append(polygon[j])
                    arcs.append(arc)
                    arc = []
                    # 设不存在 edge_sign[j] == edge_sign[j+1] == 0 的多边形，即连续的三点共线
                    while j + 1 < len(edge_sign):  # 找新弧头
                        if edge_sign[j] >= 0:
                            j -= 1
                            break
                        j += 1
                j += 1

            if arc and j == len(edge_sign):
                arc.append(polygon[0])
                arcs.append(arc)
            return arcs

        for ele in self.polygons:
            for arc in get_front_arcs(copy.deepcopy(ele)):  # 靠
                self.front_arcs.append(arc)
        # 。。。。。。。。。头疼，休息一下！！！
        return self

    def set_angle(self):
        for arc in self.front_arcs:
            for i, point in enumerate(arc):
                try:
                    arc[i].append(np.arctan2(point[1] - self.light[1], point[0] - self.light[0]))
                except AttributeError:
                    arc[i] = np.concatenate((
                        arc[i],
                        [np.arctan2(point[1] - self.light[1], point[0] - self.light[0])]), axis=0)

        # 处理多圈情况）
        '''
        for arc in self.front_arcs:
            for i, point in enumerate(arc):
                pass
        '''
        return self

    def split_arcs(self):
        # 点含义：[x, y, theta, 是否是输入的点]
        front_arcs, self.front_arcs = self.front_arcs, []
        for arc in front_arcs:
            sub_arc = []
            for i in range(len(arc)-1):
                try:
                    arc[i].append(True)
                except AttributeError:
                    arc[i] = np.append(arc[i], [True], axis=0)
                sub_arc.append(arc[i])
                s, e = arc[i], arc[i+1]
                if e[2] < 0 < s[2]:
                    div = np.array([s[0]-(e[0]-s[0])*s[1]/(e[1]-s[1]), self.light[1], np.pi, False])
                    sub_arc.append(div)
                    self.front_arcs.append(sub_arc)
                    div = copy.deepcopy(div)
                    div[2] = -div[2]
                    sub_arc = [div]
            try:
                arc[-1].append(True)
            except AttributeError:
                arc[-1] = np.append(arc[-1], [True], axis=0)
            sub_arc.append(arc[-1])
            self.front_arcs.append(sub_arc)
        pass

    def show(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        # ax.yaxis.set_major_locator(plt.MultipleLocator(1))
        ax.plot(*self.light, 'o', color='y', markersize=10)
        for ele in self.polygons:
            x, y = list(zip(*ele))
            ax.plot(x, y, 'o', color='b', markersize=5)
            ax.fill(x, y, 'r', alpha=0.3)
        for arc in self.front_arcs:
            x, y, angle, is_input = list(zip(*arc))
            ax.plot(x, y, '-', color='y', linewidth=5, alpha=0.5)
        for point in self.Lighting_points:
            ax.plot(point[0], point[1], 'o', color='y', markersize=5.2, alpha=1)

        plt.show()
        return self

    def merge_arc(self, a, b):  # 重写！！！
        # 弧不遮挡
        if a[-1][2] <= b[0][2]:
            return np.concatenate((a, b), axis=0)
            # return np.append(a, b, axis=0)

        # elif a[-1][2] > b[0][2]:
        # 弧遮挡
        if a[-1][2] < b[-1][2]:  # a的尾巴没超过b的尾巴
            b_ind = self.bi_search(a[-1], b)
            tl = self.to_left(b[b_ind], b[b_ind+1], a[-1])
            if tl < 0:  # b遮挡住a的尾巴
                a_ind = self.bi_search(b[0], a)
                jc_p = np.append(
                    self.get_intersection_point(b[0], a[a_ind], a[a_ind+1]),
                    [b[0][2], False],
                    axis=0
                )
                if b[-1][2] > a[-1][2]:
                    return np.concatenate((a[:a_ind+1], [jc_p], b), axis=0)
                    # return np.append(a[:a_ind+1], [jc_p, *b], axis=0)
            else:  # a遮挡住b的头
                jc_p = np.append(
                    self.get_intersection_point(a[-1], b[b_ind], b[b_ind + 1]),
                    [a[-1][2], False],
                    axis=0
                )
                return np.concatenate((a, [jc_p], b[b_ind + 1:]), axis=0)
                # return np.append(a, [jc_p, *b[b_ind + 1:]], axis=0)
        else:  # a[-1][2] >= b[-1][2]  a的尾巴超过b的尾巴
            a_ind = self.bi_search(b[0], a)
            tl = self.to_left(a[a_ind], a[a_ind + 1], b[0])
            if tl < 0:  # a遮挡住b的全部
                return a
            # else:  # b遮挡住了a的身体
            jc_p = np.append(
                self.get_intersection_point(b[0], a[a_ind], a[a_ind + 1]),
                [b[0][2], False],
                axis=0
            )
            a_ind2 = self.bi_search(b[-1], a)
            jc_p2 = np.append(
                self.get_intersection_point(b[0], a[a_ind2], a[a_ind2 + 1]),
                [b[0][2], False],
                axis=0
            )
            return np.concatenate((a[:a_ind+1], [jc_p], b, [jc_p2], a[a_ind2 + 1:]), axis=0)
            # return np.append(a[:a_ind+1], [jc_p, *b, jc_p2, *a[a_ind2 + 1:]], axis=0)
        pass

    def solve(self):
        front_arcs = copy.deepcopy(self.front_arcs)

        while front_arcs:
            if len(front_arcs) == 1:
                self.Lighting_arcs = front_arcs[0]
                return

            front_arcs__ = []
            for i in range(len(front_arcs)//2):
                front_arcs__.append(self.merge_arc(front_arcs[2*i], front_arcs[2*i+1]))
            if len(front_arcs) % 2 == 1:
                front_arcs__.append(front_arcs[-1])
            # del front_arcs
            front_arcs = front_arcs__

        # self.front_arcs = []
        return self


if __name__ == '__main__':
    cnt = 16
    pols = [[
        VPs.rotate(ele, 2 * np.pi * i / cnt) for ele in list(map(
            lambda t: [t[0], t[1]+5], [
                [0, 1], [1, 3], [-0.5, 4],
                [-3, 5], [-5, 4], [-0.4, 3.5]
            ]))] for i in range(cnt)
    ]
    start = time.perf_counter()
    v = VPs([0, 0], *pols)
    end = time.perf_counter()
    print(end-start)
    # print(np.array(list(map(np.array, v.front_arcs)), dtype=list))
    # print(v.front_arcs)
    # print(np.array(v.Lighting_points))
    v.show()
    '''
    print(
        VPs.rotate2(
            [1, 1],
            np.pi/4
        )
    )
    pols = [
        list(map(
            lambda t: [t[0], t[1]+2],
            [[0.25, 0.25], [0, 1], [1, 1.5], [3, 1], [2, 2], [7, 1], [1, 0]]
        )),
        [[2, 0], [4, 0], [4, -1.1], [3, -1.1]],
        [[5, -2], [5, 0], [6, 0]],
        [[np.cos(2*np.pi*i/10)-1, np.sin(2*np.pi*i/10)-1] for i in range(10)],
        [[-2, 0], [-1, 1], [0, 2], [-2, 2]],
    ]
    v = VPs([1, 1], *pols)
    print(v.front_arcs)
    print(v.Lighting_points)
    v.show()
    # 休息一下！！！
    # v.solve()
    # print(np.arctan2(-1:y, 0:x))
    '''
    pass
