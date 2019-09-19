# -*- coding:utf-8 -*-
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def a(pre, tin):
    pre = list(pre)
    tin = list(tin)
    if len(pre) == 0:
        return None
    res = TreeNode(pre[0])
    if len(pre) == 1:
        return res
    count = 0
    for i in range(len(pre)):
        if tin[i] == pre[0]:
            count = i
    res.left = a(pre[1:count + 1], tin[0:count])
    res.right = a(pre[count + 1:], tin[count + 1:])

    return res


a(pre=[1, 2, 3, 4, 5, 6, 7], tin=[3, 2, 4, 1, 6, 5, 7])
