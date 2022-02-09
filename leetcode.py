# Given the roots of two binary trees p and q, write a function to check if they are the same or not.

# def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
# #    impliment in order traversal of both trees. check if the values are the same
# #     if the values are the same keep going. else return false. return true
#         d1 = p
#         d2 = q

#         if d1 == None and d2 == None: return True
#         if d1 == None or d2 == None: return False

#         if d1.val != d2.val: return False

#         return self.isSameTree(d1.left, d2.left) and self.isSameTree(d1.right, d2.right)







# Given the root of a binary tree, return the inorder traversal of its nodes' values.

# class Solution:
#     def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:

#         if not root:
#             return []

#         def printValues(node, ans = []):

#             if not node:
#                 return []

#             printValues(node.left, ans)
#             ans.append(node.val)
#             printValues(node.right, ans)
#             return ans

#         return printValues(root)




# Given the root of a binary tree, return its maximum depth.

# A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.


# class Solution:
#     def maxDepth(self, root: Optional[TreeNode]) -> int:

#         if not root:
#             return 0

#         def addHeight(node, h = 0):
#             if not node:
#                 return h
#             h += 1
#             return max(addHeight(node.left, h), addHeight(node.right, h))


#         return addHeight(root)


# pathsum
# Given the root of a binary tree and an integer targetSum, return true if the tree has a root-to-leaf path such that adding up all the values along the path equals targetSum.

# A leaf is a node with no children.

#     def hasPathSum(self, root, sum):
#         """
#         :type root: TreeNode
#         :type sum: int
#         :rtype: bool
#         """
#         if not root: return False

#         sum -= root.val

#         if root.left == None and root.right == None:
#             return sum == 0

#         return self.hasPathSum(root.left, sum) or self.hasPathSum(root.right, sum)


# pathsum2
# Given the root of a binary tree and an integer targetSum, return all root-to-leaf paths where the sum of the node values in the path equals targetSum.
# Each path should be returned as a list of the node values, not node references.
# A root-to-leaf path is a path starting from the root and ending at any leaf node. A leaf is a node with no children.

# class Solution:
#     def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:

#         def addPath(node, nodeList, ans, summed = targetSum):

#             if not node:
#                 return []

#             summed -= node.val
#             nodeList.append(node.val)

#             if not node.left and not node.right and summed == 0:
#                 ans.append(list(nodeList))
#             else:
#                 addPath(node.left, nodeList, ans, summed)
#                 addPath(node.right, nodeList, ans, summed)

#             nodeList.pop()

#         ans = []
#         addPath(root, [], ans)
#         return ans




# 102. Binary Tree Level Order Traversal
# Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).
# class Solution:
#     def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
#         if not root:
#             return []

#         levels = []
#         def addLevel(node, level = 0):
# #             every time a recursive call is made to a child node. we increase level by 1. So we should add a new level to the levels list by appending a list.
#             if len(levels) == level:
#                 levels.append([])

#             levels[level].append(node.val)

#             if node.left:
#                 addLevel(node.left, level + 1)
#             if node.right:
#                 addLevel(node.right, level + 1)

#         addLevel(root)
#         return levels



# 200. Number of Islands
# Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.

# An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

# class Solution:
#     def numIslands(self, grid: List[List[str]]) -> int:
#         rows = len(grid)
#         cols = len(grid[0])

#         islands = 0
#         visited = set()

#         def bfs(r, c):
#             queue = collections.deque()
#             queue.append((r,c))
#             visited.add((r,c))
#             while queue:
#                 row, col = queue.popleft()
#                 directions = [[0 , 1], [ 0, -1], [1, 0], [-1, 0]]
#                 for dr, dc in directions:
#                     if (
#                     (dr + row) in range(rows) and
#                     (dc + col) in range(cols) and
#                     grid[dr + row][dc + col] == "1" and

#                     (dr + row, dc + col) not in visited):
#                         visited.add((dr + row, dc + col))
#                         queue.append((dr + row, dc + col))



# 314. Binary Tree Vertical Order Traversal
# Given the root of a binary tree, return the vertical order traversal of its nodes' values. (i.e., from top to bottom, column by column).
# If two nodes are in the same row and column, the order should be from left to right.

# class Solution:
#     def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
#         table = collections.defaultdict(list)
#         queue = collections.deque()
#         queue.append([root, 0])

#         while queue:
#             node, column = queue.popleft()
#             if node != None:
#                 table[column].append(node.val)
#                 queue.append([node.left, column - 1])
#                 queue.append([node.right, column + 1])

#         ans = [table[x] for x in sorted(table.keys())]
#         return ans

adsfasdfasdf
