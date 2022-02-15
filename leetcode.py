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

#         for r in range(rows):
#             for c in range(cols):
#                 if grid[r][c] == "1" and (r,c) not in visited:
#                     islands += 1
#                     bfs(r,c)
#         return islands



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




# 257. Binary Tree Paths
# Given the root of a binary tree, return all root-to-leaf paths in any order.

# class Solution:
#     def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
#         if not root:
#             return []

#         def helper(node, path = ""):
#             if not node:
#                 return
#             path += f"{node.val}"
#             if not node.left and not node.right:
#                 paths.append(path)
#             else:
#                 path += "->"
#                 helper(node.left, path)
#                 helper(node.right, path)


#         paths = []
#         helper(root)
#         return paths


# 199. Binary Tree Right Side View
# Given the root of a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.
# class Solution:
#     def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
#         if root == None:
#             return []
#         rightside = []
#         nextLevel = collections.deque([root])

#         while nextLevel:
#             currentLevel = nextLevel
#             nextLevel = collections.deque()

#             while currentLevel:
# #                 popping from the left gives the left most. So the last node we pop should be the right one.
#                 node = currentLevel.popleft()
#                 if node.left:
#                     nextLevel.append(node.left)
#                 if node.right:
#                     nextLevel.append(node.right)
#             rightside.append(node.val)

#         return rightside



# 286. Walls and Gates
# You are given an m x n grid rooms initialized with these three possible values:
# > -1 A wall or an obstacle.
# > 0 A gate.
# > INF Infinity means an empty room. We use the value 231 - 1 = 2147483647 to represent INF as you may assume that the distance to a gate is less than 2147483647.

# Fill each empty room with the distance to its nearest gate. If it is impossible to reach a gate, it should be filled with INF.

# class Solution:
#     def wallsAndGates(self, rooms: List[List[int]]) -> None:
#         if not rooms:
#             return

#         # helpers to identify direction and empty rooms
#         DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
#         EMPTY = 2147483647

#         # trackers for height and width of the matrix
#         h = len(rooms)
#         w = len(rooms[0])

#         # find all gates in the matrix and add them to the queue
#         q = []
#         for i in range(h):
#             for j in range(w):
#                 if rooms[i][j] == 0:
#                     q.append((i, j))

#         # BFS
#         for row, col in q:
#             # Specify the distance of all the neighbors of are current
#             # element coming out of the queue.
#             dist = rooms[row][col] + 1

#             # Check each direction from our current position
#             for dy, dx in DIRECTIONS:
#                 r = row + dy
#                 c = col + dx

#                 # if our row and column are within the bounds of our matrix AND the room is currently empty,
#                 # set the distance to the value of that position in the matrix.
#                 if 0 <= r < h and 0 <= c < w and rooms[r][c] == EMPTY:
#                     rooms[r][c] = dist
#                     q.append((r, c))


# 116. Populating Next Right Pointers in Each Node
# You are given a perfect binary tree where all leaves are on the same level, and every parent has two children. The binary tree has the following definition:
# Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.
# Initially, all next pointers are set to NULL.

# class Solution:
#     def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
#         if root == []:
#             return root

#         levels = []

#         def helper(node, level = 0):
#             if not node:
#                 return

#             if level == len(levels):
#                 levels.append([])

#             levels[level].append(node)

#             helper(node.left, level + 1)
#             helper(node.right, level + 1)

#         helper(root)

#         for level in levels:
#             for i in range(0, len(level) - 1):
#                 if i < len(level):
#                     level[i].next = level[i + 1]

#         return root



# 15. 3Sum
# Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.
# Notice that the solution set must not contain duplicate triplets.

# class Solution:
#     def threeSum(self, nums: List[int]) -> List[List[int]]:
#         nums.sort()
#         ans = []

#         def addSum(i):
#             j = i + 1
#             k = len(nums) - 1
#             while j < k:
#                 added = nums[i] + nums[j] + nums[k]
#                 if added == 0:
#                     ans.append([nums[i], nums[j], nums[k]])
#                     j += 1
#                     k -= 1
#                     while j < k and nums[j] == nums[j - 1]:
#                         j += 1
#                 elif added > 0:
#                     k -= 1
#                 else:
#                     j += 1

#         for i in range(0, len(nums) - 1):
#             if nums[i] > 0:
#                 break
#             if i == 0 or nums[i] != nums[i - 1]:
#                 addSum(i)


#         return ans


# 48. Rotate Image
# You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).
# class Solution:
#     def rotate(self, matrix: List[List[int]]) -> None:
#         n = len(matrix[0])
#         for i in range(n // 2 + n % 2):
#             for j in range(n // 2):
#                 tmp = matrix[n - 1 - j][i]
#                 matrix[n - 1 - j][i] = matrix[n - 1 - i][n - j - 1]
#                 matrix[n - 1 - i][n - j - 1] = matrix[j][n - 1 -i]
#                 matrix[j][n - 1 - i] = matrix[i][j]
#                 matrix[i][j] = tmp
