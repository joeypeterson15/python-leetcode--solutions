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




# 104. Maximum Depth of Binary Tree
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
#         self.transpose(matrix)
#         self.reflect(matrix)

#     def transpose(self, matrix):
#         n = len(matrix)
#         for i in range(n):
#             for j in range(i + 1, n):
#                 matrix[j][i], matrix[i][j] = matrix[i][j], matrix[j][i]

#     def reflect(self, matrix):
#         n = len(matrix)
#         for i in range(n):
#             for j in range(n // 2):
#                 matrix[i][j], matrix[i][-j - 1] = matrix[i][-j - 1], matrix[i][j]


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











# Allie Villarreal (she/her) to Everyone (11:43 AM)
# function getFileExtension(i) {
#  // i will be a string, but it may not have a file extension.
#  // return the file extension (with no period) if it has one, otherwise false
# }
# Allie Villarreal (she/her) to Everyone (11:44 AM)
# You have a matrix. You want to get the maximum number of coins. You can change indices of where you take a coin, but it will cost you the difference between the past index and the new index. Example:
# 1 2 3
# 1 1 3
# 1 3 1

# start off at 3. go to the next 3 (lose 0 coins), go to the final 3 (lose 1 coin). = 8
# Allie Villarreal (she/her) to Everyone (11:45 AM)
# fetch(temperatureUrl)
#  .then((response) => {
#     updateTheFancyTemperatureUI(response.json());
#  });

# temperatureUrlArray = [
# "https://.../best-data.json",
#  "https://.../average-data.json" ,
#  "https://.../worst-data.json"
# ];

# Update your app to display the best available temperature.

# assume returns error if no data back?


# there is a maze. mouse starts at any position. each intersection has n number of doors and 0 - 5 pieces of cheese.
# find the path that the mouse has to take the eat the most number of pieces of cheese where each door can lead to a new path. A dead end stops the mouse in its tracks








# 994. Rotting Oranges
# You are given an m x n grid where each cell can have one of three values:

# 0 representing an empty cell,
# 1 representing a fresh orange, or
# 2 representing a rotten orange.
# Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten.

# Return the minimum number of minutes that must elapse until no cell has a fresh orange. If this is impossible, return -1.


# class Solution:
#     def orangesRotting(self, grid: List[List[int]]) -> int:
#         rows = len(grid)
#         cols = len(grid[0])
#         FRESH = 1
#         ROTTEN = 2
#         DIRECTIONS = [(0 ,1), (0, -1), (1, 0), (-1, 0)]
#         orangesFresh = 0

#         queue = collections.deque()

#         for r in range(rows):
#             for c in range(cols):
#                 if grid[r][c] == ROTTEN:
#                     queue.append((r,c,))
#                 if grid[r][c] == FRESH:
#                     orangesFresh += 1


#         queue.append((-5, -5))
#         minutes = -1

#         while queue:
#             row,col = queue.popleft()
#             if row == -5:
#                 minutes += 1
#                 if queue:
#                     queue.append((-5, -5))

#             for dr, dc in DIRECTIONS:
#                 dx = row + dr
#                 dy = col + dc

#                 if (
#                 dx in range(rows) and
#                 dy in range(cols) and
#                 grid[dx][dy] == FRESH):
#                     grid[dx][dy] = ROTTEN
#                     queue.append((dx, dy))
#                     orangesFresh -= 1
#         return minutes if orangesFresh == 0 else -1




# 98. Validate Binary Search Tree
# Given the root of a binary tree, determine if it is a valid binary search tree (BST).
# class Solution:
#     def isValidBST(self, root: TreeNode) -> bool:

#         def validate(node, low=-math.inf, high=math.inf):
#             # Empty trees are valid BSTs.
#             if not node:
#                 return True
#             # The current node's value must be between low and high.
#             if node.val <= low or node.val >= high:
#                 return False

#             # The left and right subtree must also be valid.
#             return (validate(node.right, node.val, high) and
#                    validate(node.left, low, node.val))

#         return validate(root)


# def minimum_island(grid):
#   import collections
#   # pass # todo
#   rows = len(grid)
#   cols = len(grid[0])
#   directions = [(0,1), (0, -1), (1, 0), (-1, 0)]
#   smallest_island = float('inf')
#   visited = set()

#   def bfs(r, c):
#     count = 1
#     queue = collections.deque()
#     queue.append((r,c))
#     visited.add((r,c))
#     while queue:
#       row, col = queue.popleft()
#       for dr, dc in directions:
#         dx = dr + row
#         dy = dc + col
#         if (
#         dx in range(rows) and
#         dy in range(cols) and
#         grid[dx][dy] == 'L'and
#         (dx, dy) not in visited):
#           visited.add((dx, dy))
#           queue.append((dx, dy))
#           count += 1
#     return count


#   for r in range(rows):
#     for c in range(cols):
#       if grid[r][c] == "L" and (r,c) not in visited:
#         count_size = bfs(r,c)
#         smallest_island = min(count_size, smallest_island)

#   return smallest_island


# 54. Spiral Matrix
# Given an m x n matrix, return all elements of the matrix in spiral order.
# class Solution:
#     def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
#         # set of seen values so we do not repeat
#         visited = set()

#         # number of rows and columns in our matrix
#         rows, cols = len(matrix), len(matrix[0])

#         # r, c is the starting value
#         # dr, dc is the starting direction we want
#         r, c, dr, dc = 0, 0, 0, 1

#         # results array we will continually append to and return
#         result = []

#         # terminate loop when every cell value has been visited
#         while len(visited) < rows * cols:

#             # when we are at (r,c), append it to visited set
#             visited.add((r, c))

#             # append value at (r,c) to results array
#             result.append(matrix[r][c])

#             # if the next value is a legal value AND something we haven't seen, then visit it
#             if 0 <= r + dr < rows and 0 <= c + dc < cols and (r+dr, c+dc) not in visited:
#                 r, c = r + dr, c + dc

#             # otherwise, change directions according to rules
#             else:
#                 if (dr, dc) == (0, 1):
#                     dr, dc = 1, 0
#                 elif (dr, dc) == (1, 0):
#                     dr, dc = 0, -1
#                 elif (dr, dc) == (0, -1):
#                     dr, dc = -1, 0
#                 else:
#                     dr, dc = 0, 1

#                 # don't forget to move after changing directions!
#                 r, c = r + dr, c + dc

#         # return results array
#         return result


# 59. Spiral Matrix II
# Given a positive integer n, generate an n x n matrix filled with elements from 1 to n2 in spiral order.
# class Solution:
#     def generateMatrix(self, n: int) -> List[List[int]]:
#         # import numpy
#         # res = numpy.zeros((n,n))
#         visited = set()

#         # res = [[range(n)] for _ in range(n)]
#         res = list([[0]*n]*n)


#         s = 1

#         r = 0
#         c = 0
#         dr = 0
#         dc = 1

#         while s <= (n ** 2):

#             visited.add((r,c))

#             res[r][c] = s

#             x = r + dr
#             y = c + dc

#             if (x in range(n) and y in range(n) and (x,y) not in visited):
#                 r = x
#                 c = y

#             else:
#                 if (dr, dc) == (0, 1):
#                     dr, dc = 1, 0
#                 elif (dr, dc) == (1, 0):
#                     dr, dc = 0, -1
#                 elif (dr, dc) == (0, -1):
#                     dr, dc = -1, 0
#                 else:
#                     dr, dc = 0, 1

#                 r = r + dr
#                 c = c + dc

#             s += 1


#         return res



# 4. Median of Two Sorted Arrays
# Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.
# class Solution:
#     def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
#         m = len(nums1)
#         n = len(nums2)

#         # merged = [range(n + m)]
#         merged = [0 for _ in range(n + m)]

#         i = 0
#         j = 0
#         k = 0

#         while i < m and j < n:
#             if nums1[i] < nums2[j]:
#                 merged[k] = nums1[i]
#                 i += 1
#                 k += 1
#             else:
#                 merged[k] = nums2[j]
#                 j += 1
#                 k += 1
#         while i < m:
#             merged[k] = nums1[i]
#             k += 1
#             i += 1
#         while j < n:
#             merged[k] = nums2[j]
#             k+= 1
#             j += 1


#         m1 = merged[((n + m) // 2)]
#         if ((n + m) % 2) == 1:
#             return int(m1)
#         else:
#             m2 = merged[((n + m) // 2) - 1]
#             return int((m1 + m2) / 2)



# breadth first values
# Write a function, breadth_first_values, that takes in the root of a binary tree.
# The function should return a list containing all values of the tree in breadth-first order.

# def breadth_first_values(root):
#   pass # todo
#   import collections
#   res = []

#   if not root:
#     return res

#   queue = []
#   queue.append(root)



#   while queue:
#     node = queue.pop(0)
#     res.append(node.val)
#     if node.left:
#       queue.append((node.left))
#     if node.right:
#       queue.append((node.right))

#   return res



# structy ==> tree sum
# Write a function, tree_sum, that takes in the root of a binary tree that contains number values.
# The function should return the total sum of all values in the tree.

# def tree_sum(root):
#   # pass # todo
#     if not root:
#       return 0
#     return root.val + tree_sum(root.left) + tree_sum(root.right)




# strcuty ==> tree includes
# Write a function, tree_includes, that takes in the root of a binary tree and a target value.
# The function should return a boolean indicating whether or not the value is contained in the tree.



# def tree_includes(root, target):
#   if not root:
#     return False
#   if root.val == target:
#     return True

#   return tree_includes(root.left, target) or tree_includes(root.right, target)






# Structy ==> tree min value
# Write a function, tree_min_value, that takes in the root of a binary tree that contains number values.
# The function should return the minimum value within the tree.

# def tree_min_value(root):
#   # pass
#   if not root:
#     return float('inf')
#   if not root.left and not root.right:
#     return root.val

#   return min(root.val, min(tree_min_value(root.left), tree_min_value(root.right)))







# Structy ==> max root to leaf path sum
# Write a function, max_path_sum, that takes in the root of a binary tree that contains number values.
# The function should return the maximum sum of any root to leaf path within the tree.

# def max_path_sum(root):
#   # pass # todo
#   if not root:
#     return -float('inf')

#   if not root.left and not root.right:
#     return root.val
#   return root.val + max(max_path_sum(root.left), max_path_sum(root.right))



# reverse list

# Write a function, reverse_list, that takes in the head of a linked list as an argument.
# The function should reverse the order of the nodes in the linked list in-place and return the new head of the reversed linked list.

# def reverse_list(head):
#   # pass # todo
#   vals = []
#   curr = head
#   while curr:
#     vals.append(curr.val)
#     curr = curr.next



#   curr = head
#   c = 1
#   while curr:
#     curr.val = vals[-c]
#     curr = curr.next
#     c += 1

#   return head


# ZIPPER LIST
# Write a function, zipper_lists, that takes in the head of two linked lists as arguments.
# The function should zipper the two lists together into single linked list by alternating nodes.
# If one of the linked lists is longer than the other, the resulting list should terminate with the remaining nodes.
# The function should return the head of the zippered linked list.
# Do this in-place, by mutating the original Nodes.

# def zipper_lists(head_1, head_2):
#   # pass # todo
#   new_head = head_1
#   c1 = head_1.next
#   c2 = head_2
#   count = 0

#   while c1 and c2:
#     if count % 2 == 0:
#       new_head.next = c2
#       c2 = c2.next
#     else:
#       new_head.next = c1
#       c1 = c1.next

#     count += 1
#     new_head = new_head.next

#   if c1:
#     new_head.next = c1
#   if c2:
#     new_head.next = c2

#   return head_1





# minimum island
# Write a function, minimum_island, that takes in a grid containing Ws and Ls. W represents water and L represents land.
# The function should return the size of the smallest island.
# An island is a vertically or horizontally connected region of land.

# def minimum_island(grid):
#   import collections
#   # pass # todo

#   visited = set()
#   smallest_island = float('inf')
#   rows = len(grid)
#   cols = len(grid[0])
#   dir = [(0,1), (0,-1), (1,0), (-1,0)]

#   def bfs(r,c):
#     q = collections.deque()
#     q.append((r,c))
#     visited.add((r,c))
#     count = 1
#     while q:
#       row, col = q.popleft()
#       for dr, dc in dir:
#         dx = row + dr
#         dy = col + dc
#         if (dx in range(rows) and dy in range(cols) and (dx, dy) not in visited and grid[dx][dy] == 'L'):
#           count += 1
#           q.append((dx, dy))
#           visited.add((dx, dy))

#     return count


#   for r in range(rows):
#     for c in range(cols):
#       if grid[r][c] == 'L' and (r,c) not in visited:
#         island_size = bfs(r,c)
#         smallest_island = min(smallest_island, island_size)

#   return smallest_island



# tribonacci sequence
# Write a function tribonacci that takes in a number argument, n, and returns the n-th number of the Tribonacci sequence.
# The 0-th and 1-st numbers of the sequence are both 0.
# The 2-nd number of the sequence is 1.
# To generate further numbers of the sequence, calculate the sum of previous three numbers.
# Solve this recursively.

# def tribonacci(n):
#   tracker = {}
#   return helper(n, tracker)

# def helper(n, tracker):

#   if n == 0 or n==1:
#     return 0
#   if n == 2:
#     return 1

#   if n in tracker:
#     return tracker[n]

#   tracker[n] = helper(n - 1, tracker) + helper(n - 2, tracker) + helper(n - 3, tracker)
#   return tracker[n]



# 108. Convert Sorted Array to Binary Search Tree
# Given an integer array nums where the elements are sorted in ascending order, convert it to a height-balanced binary search tree.
# A height-balanced binary tree is a binary tree in which the depth of the two subtrees of every node never differs by more than one.


# class Solution:
#     def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:

#         def helper(left, right):

#             if left > right:
#                 return None

#             mid = (left + right) // 2

#             node = TreeNode(nums[mid])
#             node.left = helper(left, mid - 1)
#             node.right = helper(mid + 1, right)
#             return node

#         return helper(0, len(nums) - 1)

# 109. Convert Sorted List to Binary Search Tree
# Given the head of a singly linked list where elements are sorted in ascending order, convert it to a height balanced BST.
# For this problem, a height-balanced binary tree is defined as:
# a binary tree in which the depth of the two subtrees of every node never differ by more than 1.

# class Solution:
#     def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
#         vals = []
#         curr = head
#         while curr:
#             vals.append(curr.val)
#             curr = curr.next

#         def helper(left, right):
#             if left > right:
#                 return None

#             mid = (left + right) // 2
#             node = TreeNode(vals[mid])
#             node.left = helper(left, mid - 1)
#             node.right = helper(mid + 1, right)
#             return node

#         return helper(0, len(vals) - 1)



# 515. Find Largest Value in Each Tree Row
# Given the root of a binary tree, return an array of the largest value in each row of the tree (0-indexed).

# class Solution:
#     def largestValues(self, root: Optional[TreeNode]) -> List[int]:

#         levels = []

#         def helper(node, level = 0):

#             if not node:
#                 return

#             if len(levels) == level:
#                 levels.append([])

#             levels[level].append(node.val)

#             helper(node.left, level + 1)
#             helper(node.right, level + 1)

#         helper(root)

#         for i in range(len(levels)):
#             levels[i] = max(levels[i])

#         return levels




# 1730. Shortest Path to Get Food
# You are starving and you want to eat food as quickly as possible. You want to find the shortest path to arrive at any food cell.

# class Solution:
#     def getFood(self, grid: List[List[str]]) -> int:

#         rows = len(grid)
#         cols = len(grid[0])
#         directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
#         queue = collections.deque()

#         for r in range(rows):
#             for c in range(cols):
#                 if grid[r][c] == '#':
#                     queue.append((r,c))
#                     grid[r][c] = 0

#         distance = 0
#         while queue:
#             r,c = queue.popleft()

#             for dr, dc in directions:
#                 dx = dr + r
#                 dy = dc + c

#                 if dx in range(rows) and dy in range(cols):
#                     if grid[dx][dy] == 'O':
#                         distance = grid[r][c] + 1
#                         grid[dx][dy] = distance
#                         queue.append((dx,dy))

#                     if grid[dx][dy] == '*':
#                         return grid[r][c] + 1
#         return -1





# 1293. Shortest Path in a Grid with Obstacles Elimination
# You are given an m x n integer matrix grid where each cell is either 0 (empty) or 1 (obstacle).
# You can move up, down, left, or right from and to an empty cell in one step.
# Return the minimum number of steps to walk from the upper left corner (0, 0)
# to the lower right corner (m - 1, n - 1) given that you can eliminate at most k obstacles.
# If it is not possible to find such walk return -1.

# class Solution:
#     def shortestPath(self, grid: List[List[int]], k: int) -> int:

#         rows = len(grid)
#         cols = len(grid[0])
#         if rows == 1 and cols == 1:
#             return 0

#         obstacles = set()
#         visited = set()
#         directions = [(0,1), (0, -1), (1, 0), (-1, 0)]

#         for r in range(rows):
#             for c in range(cols):
#                 if grid[r][c] == 1:
#                     obstacles.add((r,c))

#         queue = collections.deque()
#         queue.append((0, 0))
#         distance = 0
#         count = 0
#         while queue:
#             row, col = queue.popleft()
#             visited.add((row, col))
#             for r, c in directions:
#                 dr = row + r
#                 dc = col + c
#                 if dr in range(rows) and dc in range(cols) and (dr, dc) not in visited:

#                     distance = grid[row][col] + 1

#                     if dr == (rows - 1) and dc == (cols - 1):
#                         return distance


#                     if grid[dr][dc] == 0:
#                         grid[dr][dc] = distance
#                         queue.append((dr, dc))

#                     if (dr, dc) in obstacles:
#                         count += 1
#                         if count <= k:
#                             queue.append((dr, dc))
#                             if grid[dr][dc] != 1:
#                                 grid[dr][dc] = min(grid[dr][dc], distance)
#                             else:
#                                 grid[dr][dc] = distance

#         return -1


# 103. Binary Tree Zigzag Level Order Traversal
# Given the root of a binary tree, return the zigzag level order traversal of its nodes' values.
# (i.e., from left to right, then right to left for the next level and alternate between).
# class Solution:
#     def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:

#         def helper(node, level = 0):
#             if not node:
#                 return

#             if level == len(levels):
#                 levels.append([])

#             levels[level].append(node.val)

#             helper(node.left, level + 1)
#             helper(node.right, level + 1)

#         levels = []
#         helper(root)

#         for i in range(len(levels)):
#             if i % 2 == 1:
#                 levels[i] = levels[i][::-1]


#         return levels



# 1602. Find Nearest Right Node in Binary Tree
# Given the root of a binary tree and a node u in the tree,
# return the nearest node on the same level that is to the right of u, or return null if u is the rightmost node in its level.
# class Solution:
#     def findNearestRightNode(self, root: TreeNode, u: TreeNode) -> Optional[TreeNode]:

#         levels = []

#         def bfs(node, level = 0):

#             if not node:
#                 return

#             if len(levels) == level:
#                 levels.append([])

#             # if len(levels[level]) > 0 and levels[level][-1] == u.val:
#             #     return node

#             levels[level].append(node)

#             bfs(node.left, level + 1)
#             bfs(node.right, level + 1)


#         bfs(root)

#         for i in range(len(levels)):
#             for j in range(len(levels[i])):
#                 if levels[i][j] == u and (j + 1) in range(len(levels[i])):
#                     return levels[i][j + 1]
#         return None








# 235. Lowest Common Ancestor of a Binary Search Tree
# Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.
# According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between
# two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”
# class Solution:
#     def lowestCommonAncestor(self, root, p, q):
#         """
#         :type root: TreeNode
#         :type p: TreeNode
#         :type q: TreeNode
#         :rtype: TreeNode
#         """
#         # Value of current node or parent node.
#         parent_val = root.val



#         # If both p and q are greater than parent
#         if p.val > parent_val and q.val > parent_val:
#             return self.lowestCommonAncestor(root.right, p, q)
#         # If both p and q are lesser than parent
#         elif p.val < parent_val and q.val < parent_val:
#             return self.lowestCommonAncestor(root.left, p, q)
#         # We have found the split point, i.e. the LCA node.
#         else:
#             return root





# 236. Lowest Common Ancestor of a Binary Tree
# Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

# According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes
# p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”
# class Solution:
#     def __init__(self):
#         self.ans = None

#     def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':


#         def helper(node):

#             if not node:
#                 return False

#             left = helper(node.left)
#             right = helper(node.right)
#             mid = node == p or node == q

#             if left + right + mid >= 2:
#                 self.ans = node

#             return mid or left or right

#         helper(root)
#         return self.ans


# 1644. Lowest Common Ancestor of a Binary Tree II
# Given the root of a binary tree, return the lowest common ancestor (LCA) of two given nodes, p and q. If either node p or q does not exist in the tree, return null. All values of the nodes in the tree are unique.

# According to the definition of LCA on Wikipedia:
# "The lowest common ancestor of two nodes p and q in a binary tree T is the lowest node
# that has both p and q as descendants (where we allow a node to be a descendant of itself)".
# A descendant of a node x is a node y that is on the path from node x to some leaf node.

# class Solution:
#     def __init__(self):
#         self.ans = None

#     def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':

#         def helper(node):

#             if not node:
#                 return False

#             left = helper(node.left)
#             right = helper(node.right)
#             mid = node == p or node == q

#             if mid + left + right >= 2:
#                 self.ans = node

#             return mid or left or right

#         helper(root)
#         return self.ans



# 1650. Lowest Common Ancestor of a Binary Tree III
# Given two nodes of a binary tree p and q, return their lowest common ancestor (LCA).
# class Solution:
#     def lowestCommonAncestor(self, p: 'Node', q: 'Node') -> 'Node':
#         p_set = set()
#         while p:
#             p_set.add(p)
#             p = p.parent

#         while q:
#             if q in p_set:
#                 return q
#             q = q.parent
#         return None




# 129. Sum Root to Leaf Numbers
# You are given the root of a binary tree containing digits from 0 to 9 only.

# Each root-to-leaf path in the tree represents a number.

# For example, the root-to-leaf path 1 -> 2 -> 3 represents the number 123.
# Return the total sum of all root-to-leaf numbers. Test cases are generated so that the answer will fit in a 32-bit integer.

# A leaf node is a node with no children.

# # 543. Diameter of Binary Tree
# # Given the root of a binary tree, return the length of the diameter of the tree.

# # The diameter of a binary tree is the length of the longest path between any two nodes in a tree.
# # This path may or may not pass through the root.
# # class Solution:
# #     def diameterOfBinaryTree(self, root: TreeNode) -> int:
# #         diameter = 0

# #         def longest_path(node):
# #             if not node:
# #                 return 0
# #             nonlocal diameter
# #             # recursively find the longest path in
# #             # both left child and right child
# #             left_path = longest_path(node.left)
# #             right_path = longest_path(node.right)

# #             # update the diameter if left_path plus right_path is larger
# #             diameter = max(diameter, left_path + right_path)

# #             # return the longest one between left_path and right_path;
# #             # remember to add 1 for the path connecting the node and its parent
# #             return max(left_path, right_path) + 1

# #         longest_path(root)
# #         return diameter






# 988. Smallest String Starting From Leaf
# You are given the root of a binary tree where each node has a value in the range [0, 25] representing the letters 'a' to 'z'.

# Return the lexicographically smallest string that starts at a leaf of this tree and ends at the root.

# As a reminder, any shorter prefix of a string is lexicographically smaller.
# class Solution(object):
#     def smallestFromLeaf(self, root):
#         self.ans = "~"

#         def dfs(node, A):
#             if node:
#                 A.append(chr(node.val + ord('a')))
#                 if not node.left and not node.right:
#                     self.ans = min(self.ans, "".join(reversed(A)))

#                 dfs(node.left, A)
#                 dfs(node.right, A)
#                 A.pop()

#         dfs(root, [])
#         return self.ans



# 124. Binary Tree Maximum Path Sum
# A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them.
# A node can only appear in the sequence at most once. Note that the path does not need to pass through the root.
# The path sum of a path is the sum of the node's values in the path.
# Given the root of a binary tree, return the maximum path sum of any non-empty path.
# class Solution:
#     def __init__(self):
#         self.max_length = -float('inf')

#     def maxPathSum(self, root: Optional[TreeNode]) -> int:


#         def dfs(node):

#             if not node:
#                 return 0

#             left_path = max(dfs(node.left), 0)
#             right_path = max(dfs(node.right), 0)

#             curr_sum_paths = node.val + left_path + right_path

#             self.max_length = max(self.max_length, curr_sum_paths)

#             return node.val + max(left_path, right_path)


#         dfs(root)
#         return self.max_length



# 547. Number of Provinces
# class Solution:
#     def findCircleNum(self, isConnected: List[List[int]]) -> int:

#         l = len(isConnected)
#         provinces = 0

#         adj_list = {i : [] for i in range(l)}

#         for i in range(l):
#             for j in range(l):
#                 if isConnected[i][j] == 1:
#                     adj_list[i].append(j)
#                     adj_list[j].append(i)


#         visited = set()

#         def dfs(n, prev):
#             if n in visited:
#                 return

#             visited.add(n)

#             for neighbor in adj_list[n]:
#                 if neighbor == prev:
#                     continue
#                 dfs(neighbor, n)

#         for i in range(l):
#             if i not in visited:
#                 provinces += 1
#                 dfs(i, -1)

#         return provinces



# 323. Number of Connected Components in an Undirected Graph
# class Solution:
#     def countComponents(self, n: int, edges: List[List[int]]) -> int:
#         if not n:
#             return 0

#         components = 0

#         adj_list = {i : [] for i in range(n)}
#         for a,b in edges:
#             adj_list[a].append(b)
#             adj_list[b].append(a)

#         visited = set()

#         def dfs(node, prev):

#             if node in visited:
#                 return

#             visited.add(node)

#             for n in adj_list[node]:
#                 if n == prev:
#                     continue
#                 dfs(n, node)


#         for i in range(n):
#             if i not in visited:
#                 components += 1
#                 dfs(i, -1)

#         return components




# 112. Path Sum
# class Solution:
#     def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:

#         def dfs(node, sum):
#             if not node:
#                 return False

#             sum += node.val

#             if not node.left and not node.right:
#                 if sum != targetSum:
#                     return False
#                 else:
#                     return True

#             return dfs(node.left, sum) or dfs(node.right, sum)

#         return dfs(root, 0)


# 366. Find Leaves of Binary Tree
# class Solution:
#     def findLeaves(self, root: Optional[TreeNode]) -> List[List[int]]:

#         res = []

#         def getHeight(node):
#             if not node:
#                 return -1

#             left_height = getHeight(node.left)
#             right_height = getHeight(node.right)

#             curr_height = max(left_height, right_height) + 1

#             if curr_height == len(res):
#                 res.append([])

#             res[curr_height].append(node.val)
#             return curr_height

#         getHeight(root)
#         return res


# 394. Decode String
# class Solution:
#     def decodeString(self, s: str) -> str:

#         stack = collections.deque()

#         for c in s:
#             if c != ']':
#                 stack.append(c)
#             else:
#                 curr_str = ''
#                 while stack[len(stack) - 1] != '[':
#                     curr_str += stack.pop()

#                 stack.pop()
#                 rev = curr_str[::-1]

#                 k = ''
#                 while stack and stack[len(stack) - 1].isnumeric():
#                     k += stack.pop()
#                 k = int(k[::-1])

#                 while k > 0:
#                     k -= 1
#                     for i in rev:
#                         stack.append(i)

#         return ''.join(stack)


# 1026. Maximum Difference Between Node and Ancestor
# class Solution:
#     def __init__(self):
#         self.max_diff = -float('inf')
#     def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:

#         def helper(node, minim, maxim):
#             if not node:
#                 return
#             if node.val < minim:
#                 minim = node.val
#             if node.val > maxim:
#                 maxim = node.val
#             self.max_diff = max((maxim - minim), self.max_diff)

#             helper(node.left, minim, maxim)
#             helper(node.right, minim, maxim)

#         helper(root, float('inf'), -float('inf'))
#         return self.max_diff


# 652. Find Duplicate Subtrees
# class Solution:
#        def findDuplicateSubtrees(self, root: Optional[TreeNode]) -> list[Optional[TreeNode]]:

#             result = []
#             paths = defaultdict(int)

#             def get_path(node):
#                 if not node:
#                     return "None"
#                 else:
#                     path = str(node.val)
#                 path += '.' + get_path(node.left)
#                 path += '.' + get_path(node.right)
#                 paths[path] += 1
#                 if paths[path] == 2:
#                     result.append(node)
#                 return path

#             get_path(root)
#             return result


# 2096. Step-By-Step Directions From a Binary Tree Node to Another
# class Solution(object):
# 	def getDirections(self, root, startValue, destValue):
# 		"""
# 		:type root: Optional[TreeNode]
# 		:type startValue: int
# 		:type destValue: int
# 		:rtype: str
# 		"""

# 		def lca(root):
# 			if root == None:
#               return None
#
#           if root.val == startValue or root.val == destValue:
# 				return root

# 			left = lca(root.left)
# 			right = lca(root.right)

# 			if left and right:
# 				return root

# 			return left or right

# 		def dfs(root, value, path):
# 			if root is None:
# 				return False

# 			if root.val == value:
# 				return True

# 			if dfs(root.left, value, path):
# 				path.append("L")
# 				return True

# 			elif dfs(root.right, value, path):
# 				path.append("R")
# 				return True

# 			return False

# 		root = lca(root)
# 		start_to_root = []
# 		dfs(root, startValue, start_to_root)

# 		dest_to_root = []
# 		dfs(root, destValue, dest_to_root)

# 		return "U" * len(start_to_root) + ''.join(reversed(dest_to_root))


# Lowest Common Ancestor IV
# class Solution:
#     def lowestCommonAncestor(self, root: 'TreeNode', nodes: 'List[TreeNode]') -> 'TreeNode':

#         def helper(node):
#             if not node:
#                 return None
#             for n in nodes:
#                 if n.val == node.val:
#                     return node

#             left = helper(node.left)
#             right = helper(node.right)

#             if left and right:
#                 return node

#             return left or right

#         return helper(root)


# 1740. Find Distance in a Binary Tree
# class Solution:
#     def findDistance(self, root: Optional[TreeNode], p: int, q: int) -> int:

#         def lca(node):
#             if not node:
#                 return None

#             if node.val == p or node.val == q:
#                 return node

#             left = lca(node.left)
#             right = lca(node.right)

#             if left and right:
#                 return node

#             return left or right


#         def dfs(node, target, distance):
#             if not node:
#                 return False
#             if node.val == target:
#                 return True

#             if dfs(node.left, target, distance):
#                 distance.append(node.val)
#                 return True

#             if dfs(node.right,target, distance):
#                 distance.append(node.val)
#                 return True

#             return False


#         lca_node = lca(root)

#         left_dist = []
#         dfs(lca_node, p, left_dist)

#         right_dist = []
#         dfs(lca_node, q, right_dist)

#         return len(left_dist) + len(right_dist)





# 1469. Find All The Lonely Nodes
# class Solution:
#     def getLonelyNodes(self, root: Optional[TreeNode]) -> List[int]:

#         def helper(node, res):
#             if not node:
#                 return

#             if not node.left and not node.right:
#                 return

#             if not node.left:
#                 res.append(node.right.val)

#             if not node.right:
#                 res.append(node.left.val)

#             helper(node.left, res)
#             helper(node.right, res)


#         result = []
#         helper(root, result)
#         return result





# 965. Univalued Binary Tree
# class Solution:
#     def isUnivalTree(self, root: Optional[TreeNode]) -> bool:

#         uni_val = root.val

#         def helper(node):
#             if not node:
#                 return True

#             if node.val != uni_val:
#                 return False

#             return helper(node.left) and helper(node.right)

#         return helper(root)




# 333. Largest BST Subtree
# class Solution:
#     def is_valid_bst(self, root: Optional[TreeNode]) -> bool:
#         """Check if given tree is a valid BST using in-order traversal."""
#         # An empty tree is a valid Binary Search Tree.
#         if not root:
#             return True

#         # If left subtree is not a valid BST return false.
#         if not self.is_valid_bst(root.left):
#             return False

#         # If current node's value is not greater than the previous
#         # node's value in the in-order traversal return false.
#         if self.previous and self.previous.val >= root.val:
#             return False

#         # Update previous node to current node.
#         self.previous = root

#         # If right subtree is not a valid BST return false.
#         return self.is_valid_bst(root.right)

#     # Count nodes in current tree.
#     def count_nodes(self, root: Optional[TreeNode]) -> int:
#         if not root:
#             return 0

#         # Add nodes in left and right subtree.
#         # Add 1 and return total size.
#         return 1 + self.count_nodes(root.left) + self.count_nodes(root.right)

#     def largestBSTSubtree(self, root: Optional[TreeNode]) -> int:
#         if not root:
#             return 0

#         # Previous node is initially null.
#         self.previous = None

#         # If current subtree is a validBST, its children will have smaller size BST.
#         if self.is_valid_bst(root):
#             return self.count_nodes(root)

#         # Find BST in left and right subtrees of current nodes.
#         return max(self.largestBSTSubtree(root.left), self.largestBSTSubtree(root.right))



# 404. Sum of Left Leaves
# class Solution:
#     def __init__(self):
#         self.sum = 0

#     def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:

#         if not root:
#             return 0

#         def helper(node, isleft):

#             if not node.left and not node.right:
#                 if isleft:
#                     return node.val
#                 else:
#                     return 0
#             total = 0

#             if node.left:
#                 total += helper(node.left, True)
#             if node.right:
#                 total += helper(node.right, False)

#             return total

#         return helper(root, False)




# 572. Subtree of Another Tree
# class Solution:

#     def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:


#         def dfs(curr_node, sub_node):
#             if not curr_node and not sub_node:
#                 return True

#             if (not curr_node and sub_node) or (not sub_node and curr_node):
#                 return False

#             if curr_node.val != sub_node.val:
#                 return False

#             left = dfs(curr_node.left, sub_node.left)
#             right = dfs(curr_node.right, sub_node.right)
#             return left and right


#         def bfs(node, subNode):
#             if not node:
#                 return False

#             queue = collections.deque()
#             queue.append(node)

#             while queue:
#                 curr_node = queue.popleft()
#                 if curr_node.val == subNode.val:
#                     isSame = dfs(curr_node, subNode)
#                     if isSame:
#                         return True
#                 if curr_node.left:
#                     queue.append(curr_node.left)
#                 if curr_node.right:
#                     queue.append(curr_node.right)

#             return False

#         return bfs(root, subRoot)





# 508. Most Frequent Subtree Sum
# class Solution:
#     def findFrequentTreeSum(self, root: Optional[TreeNode]) -> List[int]:

#         hashMap = collections.defaultdict(int)

#         def helper(node):
#             if not node:
#                 return 0

#             left = helper(node.left)
#             right = helper(node.right)

#             curr_sum = left + right + node.val
#             hashMap[curr_sum] += 1
#             return curr_sum

#         helper(root)
#         res = []
#         max_count = max(hashMap.values())
#         return [val for val in hashMap.keys() if hashMap[val] == max_count]



# 1973. Count Nodes Equal to Sum of Descendants
# class Solution:
#     def __init__(self):
#         self.count = 0

#     def equalToDescendants(self, root: Optional[TreeNode]) -> int:

#         def helper(node):

#             if not node:
#                 return 0

#             left = helper(node.left)
#             right = helper(node.right)

#             if left + right == node.val:
#                 self.count += 1

#             return left + right + node.val

#         helper(root)
#         return self.count





# 1120. Maximum Average Subtree
# class Solution:
#     def __init__(self):
#         self.count = 0
#     def maximumAverageSubtree(self, root: Optional[TreeNode]) -> float:

#         def countNodes(node):
#             if not node:
#                 return 0
#             left = countNodes(node.left)
#             right = countNodes(node.right)
#             return left + right + 1

#         def findSum(node):
#             if not node:
#                 return 0

#             left = findSum(node.left)
#             right = findSum(node.right)

#             curr_sum = left + right + node.val
#             num_nodes = countNodes(node)
#             averages.append(curr_sum/num_nodes)

#             return curr_sum

#         averages = []
#         findSum(root)
#         return max(averages)





# !!!!NOT THE CORRECT SOLUTION!!!
# 250. Count Univalue Subtrees

#     def countUnivalSubtrees(self, root: Optional[TreeNode]) -> int:

#         def bfs(uni, node):
#             if not node:
#                 return True
#             if uni != node.val:
#                 return False
#             left = bfs(uni, node.left)
#             right = bfs(uni, node.right)
#             return left and right



#         def helper(node):
#             if not node:
#                 return -1

#             if not node.left and not node.right:
#                 self.count += 1
#                 return node.val

#             left = helper(node.left)
#             right = helper(node.right)

#             if left == right:
#                 if bfs(left, node.left) and bfs(right, node.right):
#                     self.count += 1

#             return node.val

#         helper(root)
#         return self.count



# 56. Merge Intervals
# class Solution:
#     def merge(self, intervals: List[List[int]]) -> List[List[int]]:

#         intervals.sort(key=lambda x: x[0])
#         res = []

#         for interval in intervals:
#             if not res or res[-1][1] < interval[0]:
#                 res.append(interval)
#             else:
#                 res[-1][1] = max(interval[1], res[-1][1])

#         return res



# Meeting Rooms
# class Solution:
#     def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
# #         need to find if any of the intervals overlap. If any overlap return false

#         intervals.sort(key=lambda i: i[0])
#         # res = []

#         for i in range(len(intervals)):
#             if (i + 1) in range(len(intervals)) and intervals[i][1] > intervals[i + 1][0]:
#                 return False

#         return True



# 253. Meeting Rooms II
# class Solution:
#     def minMeetingRooms(self, intervals: List[List[int]]) -> int:

#         # If there is no meeting to schedule then no room needs to be allocated.
#         if not intervals:
#             return 0

#         # The heap initialization
#         free_rooms = []

#         # Sort the meetings in increasing order of their start time.
#         intervals.sort(key= lambda x: x[0])

#         # Add the first meeting. We have to give a new room to the first meeting.
#         heapq.heappush(free_rooms, intervals[0][1])

#         # For all the remaining meeting rooms
#         for i in intervals[1:]:

#             # If the room due to free up the earliest is free, assign that room to this meeting.
#             if free_rooms[0] <= i[0]:
#                 heapq.heappop(free_rooms)

#             # If a new room is to be assigned, then also we add to the heap,
#             # If an old room is allocated, then also we have to add to the heap with updated end time.
#             heapq.heappush(free_rooms, i[1])

#         # The size of the heap tells us the minimum rooms required for all the meetings.
#         return len(free_rooms)


# 7. Reverse Integer
# class Solution:
#     def reverse(self, x: int) -> int:
#         if x == 0:
#             return 0

#         string = str(x)

#         isNeg = False

#         if x < 0:
#             isNeg = True
#             string = string[1:]

#         revStr = string[::-1]
#         while revStr[0] == '0':
#             revStr = revStr[1:]
#         if isNeg:
#             revStr = '-' + revStr

#         newNum = int(revStr)
#         if newNum > 2 ** 31 or newNum < -2**31:
#             return 0
#         return newNum


# 244. Shortest Word Distance II
# from collections import defaultdict
# class WordDistance:

#     def __init__(self, words):
#         """
#         :type words: List[str]
#         """
#         self.locations = defaultdict(list)

#         # Prepare a mapping from a word to all it's locations (indices).
#         for i, w in enumerate(words):
#             self.locations[w].append(i)

#     def shortest(self, word1, word2):
#         """
#         :type word1: str
#         :type word2: str
#         :rtype: int
#         """
#         loc1, loc2 = self.locations[word1], self.locations[word2]
#         l1, l2 = 0, 0
#         min_diff = float("inf")

#         # Until the shorter of the two lists is processed
#         while l1 < len(loc1) and l2 < len(loc2):
#             min_diff = min(min_diff, abs(loc1[l1] - loc2[l2]))
#             if loc1[l1] < loc2[l2]:
#                 l1 += 1
#             else:
#                 l2 += 1
#         return min_diff


# 63. Unique Paths II
# class Solution(object):
#     def uniquePathsWithObstacles(self, obstacleGrid):
#         """
#         :type obstacleGrid: List[List[int]]
#         :rtype: int
#         """

#         m = len(obstacleGrid)
#         n = len(obstacleGrid[0])

#         # If the starting cell has an obstacle, then simply return as there would be
#         # no paths to the destination.
#         if obstacleGrid[0][0] == 1:
#             return 0

#         # Number of ways of reaching the starting cell = 1.
#         obstacleGrid[0][0] = 1

#         # Filling the values for the first column
#         for i in range(1,m):
#             obstacleGrid[i][0] = int(obstacleGrid[i][0] == 0 and obstacleGrid[i-1][0] == 1)

#         # Filling the values for the first row
#         for j in range(1, n):
#             obstacleGrid[0][j] = int(obstacleGrid[0][j] == 0 and obstacleGrid[0][j-1] == 1)

#         # Starting from cell(1,1) fill up the values
#         # No. of ways of reaching cell[i][j] = cell[i - 1][j] + cell[i][j - 1]
#         # i.e. From above and left.
#         for i in range(1,m):
#             for j in range(1,n):
#                 if obstacleGrid[i][j] == 0:
#                     obstacleGrid[i][j] = obstacleGrid[i-1][j] + obstacleGrid[i][j-1]
#                 else:
#                     obstacleGrid[i][j] = 0

#         # Return value stored in rightmost bottommost cell. That is the destination.
#         return obstacleGrid[m-1][n-1]




# 881. Boats to Save People
# class Solution:
#     def numRescueBoats(self, people: List[int], limit: int) -> int:
#         people = sorted(people)
#         lo = 0
#         hi = len(people)-1
#         boats = 0
#         while lo <= hi:
#             if people[lo] + people[hi] <= limit:
#                 lo += 1
#                 hi -= 1
#             else:
#                 hi -= 1
#             boats += 1
#         return boats





# 1099. Two Sum Less Than K
# class Solution:
#     def twoSumLessThanK(self, nums: List[int], k: int) -> int:
#         ans = -1
#         nums.sort()
#         lo = 0
#         hi = len(nums) - 1

#         while lo < hi:
#             curr_sum = nums[lo] + nums[hi]
#             if curr_sum < k:
#                 ans = max(ans, curr_sum)
#                 lo += 1
#             else:
#                 hi -= 1

#         return ans



# 259. 3Sum Smaller
# class Solution:
#     def threeSumSmaller(self, nums: List[int], target: int) -> int:
#         nums.sort()
#         num_sums = 0

#         def findTwoSums(nums, lo, new_target):
#             sum = 0
#             hi = len(nums) - 1
#             while lo <= hi:
#                 if nums[lo] + nums[hi] < new_target:
#                     sum += hi - lo
#                     lo += 1
#                 else:
#                     hi -= 1
#             return sum


#         for i in range(len(nums) - 2):
#             lo = i + 1
#             new_target = target - nums[i]
#             curr_sum = findTwoSums(nums, lo, new_target)
#             num_sums += curr_sum
#         return num_sums




# 16. 3Sum Closest
# class Solution:
#     def threeSumClosest(self, nums: List[int], target: int) -> int:
#         diff = float('inf')
#         nums.sort()

#         for i in range(len(nums)):
#             lo = i + 1
#             hi = len(nums) - 1
#             while lo < hi:
#                 curr_sum = nums[i] + nums[lo] + nums[hi]
#                 if abs(target - curr_sum) < abs(diff):
#                     diff = target - curr_sum
#                 if curr_sum < target:
#                     lo += 1
#                 else:
#                     hi -= 1
#                 if diff == 0:
#                     break

#         return target - diff




# 1257. Smallest Common Region
# class Solution:
#     def __init__(self):
#         self.ans = None
#         self.hash_regions = None
#     def findSmallestRegion(self, regions: List[List[str]], region1: str, region2: str) -> str:
#         self.hash_regions = defaultdict(list)

#         for region in regions:
#             self.hash_regions[region[0]] = region[1:]

#         def findLowestCommonRegion(reg, region1, region2):
#             if not reg:
#                 return False

#             check = []
#             for region in self.hash_regions[reg]:
#                 check.append(findLowestCommonRegion(region, region1, region2))

#             curr_region = reg == region1 or reg == region2

#             if curr_region + sum(check) >= 2:
#                 self.ans = reg

#             return curr_region or any(check)

#         findLowestCommonRegion(regions[0][0], region1, region2)
#         return self.ans



# 34. Find First and Last Position of Element in Sorted Array
# class Solution:
#     def searchRange(self, nums: List[int], target: int) -> List[int]:

#         lo = 0
#         hi = len(nums) - 1
#         res = []
#         while lo <= hi:
#             mid = (lo + hi) // 2

#             if nums[mid] < target:
#                 lo = mid + 1
#             if nums[mid] > target:
#                 hi = mid - 1
#             if nums[mid] == target:
#                 lo = mid
#                 hi = mid
#                 while lo - 1 in range(len(nums)) and nums[lo - 1] == target:
#                     lo -= 1
#                 res.insert(1, lo)
#                 while hi + 1 in range(len(nums)) and nums[hi + 1] == target:
#                     hi += 1
#                 res.append(hi)
#                 return res
#         return [-1, -1]



# 33. Search in Rotated Sorted Array
# class Solution:
#     def search(self, nums: List[int], target: int) -> int:
#         lo = 0
#         hi = len(nums) - 1
#         inBeginRange = False
#         if target >= nums[0]:
#             inBeginRange = True


#         while lo <= hi:
#             if inBeginRange:
#                 while nums[hi] < nums[lo]:
#                     hi -= 1
#             else:
#                 while nums[hi] < nums[lo]:
#                     lo += 1

#             mid = (lo + hi) // 2

#             if nums[mid] > target:
#                 hi = mid - 1
#             elif nums[mid] < target:
#                 lo = mid + 1
#             else:
#                 return mid

#         return -1


# 74. Search a 2D Matrix
# class Solution:
#     def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
#         rows = len(matrix)
#         cols = len(matrix[0])

#         for r in range(rows):
#             if matrix[r][0] <= target <= matrix[r][cols - 1]:
#                 lo = 0
#                 hi = cols - 1
#                 while lo <= hi:
#                     mid = (lo + hi) // 2
#                     if matrix[r][mid] < target:
#                         lo = mid + 1
#                     elif matrix[r][mid] > target:
#                         hi = mid - 1
#                     else:
#                         return True
#         return False



# 2089. Find Target Indices After Sorting Array
# class Solution:
#     def targetIndices(self, nums: List[int], target: int) -> List[int]:
#         nums.sort()
#         res = []
#         for i, n in enumerate(nums):
#             if n == target:
#                 res.append(i)
#         return res




# 1331. Rank Transform of an Array
# class Solution:
#     def arrayRankTransform(self, arr: List[int]) -> List[int]:

#         length = len(arr)

#         trackIndex = defaultdict(set)

#         res = [None for _ in range(length)]

#         for i in range(length):
#             trackIndex[arr[i]].add(i)

#         arr.sort()
#         rank = 1
#         i = 0
#         while i < length:
#             for j in trackIndex[arr[i]]:
#                 res[j] = rank
#             while (i + 1) in range(length) and arr[i + 1] == arr[i]:
#                 i += 1
#             i += 1
#             rank += 1

#         return res


# 136. Single Number
# class Solution:
#     def singleNumber(self, nums: List[int]) -> int:
#         counter = defaultdict(int)
#         for n in nums:
#             counter[n] += 1

#         for k, v in counter.items():
#             if v == 1:
#                 return k




# 169. Majority Element
# class Solution:
#     def majorityElement(self, nums: List[int]) -> int:
#         counter = defaultdict(int)
#         length = len(nums)
#         for n in nums:
#             counter[n] += 1
#             if counter[n] > length // 2:
#                 return n


# 27. Remove Element
# class Solution:
#     def removeElement(self, nums: List[int], val: int) -> int:
#         i = 0
#         length = len(nums)
#         for j in range(length):
#             if nums[j] != val:
#                 nums[i] = nums[j]
#                 i += 1

#         return i



# 203. Remove Linked List Elements
# # Definition for singly-linked list.
# # class ListNode:
# #     def __init__(self, val=0, next=None):
# #         self.val = val
# #         self.next = next
# class Solution:
#     def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
#         sentinal = ListNode(0)
#         sentinal.next = head

#         curr = head
#         prev = sentinal

#         while curr:
#             if curr.val == val:
#                 prev.next = curr.next
#             else:
#                 prev = curr
#             curr = curr.next

#         return sentinal.next





# 237. Delete Node in a Linked List
# class Solution:
#     def deleteNode(self, node):
#         """
#         :type node: ListNode
#         :rtype: void Do not return anything, modify node in-place instead.
#         """
#         node.val=node.next.val
#         node.next=node.next.next




# 57. Insert Interval
# class Solution:
#     def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
#         res = []
#         intervals.append(newInterval)
#         intervals.sort(key=lambda i:i[0])
#         res.append(intervals[0])
#         for i in range(1, len(intervals)):
#             if intervals[i][0] <= res[-1][1]:
#                 res[-1][1] = max(res[-1][1], intervals[i][1])
#             else:
#                 res.append(intervals[i])
#         return res



# 153. Find Minimum in Rotated Sorted Array
# class Solution(object):
#     def findMin(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: int
#         """
#         # If the list has just one element then return that element.
#         if len(nums) == 1:
#             return nums[0]

#         # left pointer
#         left = 0
#         # right pointer
#         right = len(nums) - 1

#         # if the last element is greater than the first element then there is no rotation.
#         # e.g. 1 < 2 < 3 < 4 < 5 < 7. Already sorted array.
#         # Hence the smallest element is first element. A[0]
#         if nums[right] > nums[0]:
#             return nums[0]

#         # Binary search way
#         while right >= left:
#             # Find the mid element
#             mid = (left + right) // 2
#             # if the mid element is greater than its next element then mid+1 element is the smallest
#             # This point would be the point of change. From higher to lower value.
#             if nums[mid] > nums[mid + 1]:
#                 return nums[mid + 1]
#             # if the mid element is lesser than its previous element then mid element is the smallest
#             if nums[mid - 1] > nums[mid]:
#                 return nums[mid]

#             # if the mid elements value is greater than the 0th element this means
#             # the least value is still somewhere to the right as we are still dealing with elements greater than nums[0]
#             if nums[mid] > nums[0]:
#                 left = mid + 1
#             # if nums[0] is greater than the mid value then this means the smallest value is somewhere to the left
#             else:
#                 right = mid - 1



# 162. Find Peak Element
# class Solution:
#     def findPeakElement(self, nums: List[int]) -> int:
#         length = len(nums)
#         for i in range(length - 1):
#             if nums[i + 1] < nums[i]:
#                 return i

#         return length - 1




# 119. Pascal's Triangle II
# class Solution:
#     def getRow(self, rowIndex: int) -> List[int]:
#         if rowIndex == 0:
#             return [1]
#         if rowIndex == 1:
#             return [1,1]
#         prevRow = self.getRow(rowIndex - 1)
#         currRow = [1]
#         for i in range(len(prevRow) - 1):
#             currRow.append(prevRow[i] + prevRow[i + 1])
#         currRow.append(1)
#         return currRow


# 118. Pascal's Triangle
# class Solution:
#     def generate(self, numRows: int) -> List[List[int]]:
#         res = [[1], [1,1]]
#         if numRows == 1:
#             return [res[0]]
#         if numRows == 2:
#             return res

#         for i in range(2, numRows):
#             curr_row = [1]
#             for j in range(len(res[i - 1]) - 1):
#                 sum = res[i - 1][j] + res[i - 1][j + 1]
#                 curr_row.append(sum)
#             curr_row.append(1)
#             res.append(curr_row)
#         return res





# 1460. Make Two Arrays Equal by Reversing Sub-arrays
# class Solution:
#     def canBeEqual(self, target: List[int], arr: List[int]) -> bool:
#         target.sort()
#         arr.sort()

#         return target == arr




# 1836. Remove Duplicates From an Unsorted Linked List
# class Solution:
#     def deleteDuplicatesUnsorted(self, head: ListNode) -> ListNode:
#         counter = defaultdict(int)
#         curr = head
#         while curr:
#             counter[curr.val] += 1
#             curr = curr.next

#         sentinel = ListNode(0, head)
#         prev = sentinel
#         curr = head
#         while curr:
#             if counter[curr.val] >= 2:
#                 prev.next = curr.next
#             else:
#                 prev = curr
#             curr = curr.next

#         return sentinel.next




# 83. Remove Duplicates from Sorted List
# class Solution:
#     def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
#         sentinel = ListNode(0, head)
#         curr = head
#         prev = sentinel
#         while curr:
#             if curr.next and curr.next.val == curr.val:
#                 while curr.next and curr.next.val == curr.val:
#                     prev.next = curr.next
#                     curr = curr.next
#             else:
#                 prev = curr
#                 curr = curr.next
#         return sentinel.next





# 844. Backspace String Compare
# class Solution:
#     def backspaceCompare(self, s: str, t: str) -> bool:

#         def backSpace(string):
#             stack = []
#             for s in string:
#                 if s != '#':
#                     stack.append(s)
#                 else:
#                     if len(stack) > 0:
#                         stack.pop()
#             return "".join(stack)

#         new_str1 = backSpace(s)
#         new_str2 = backSpace(t)

#         return new_str1 == new_str2



# 240. Search a 2D Matrix II
# class Solution:
#     def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
#         rows = len(matrix)
#         cols = len(matrix[0])

#         for i in range(rows):

#             if matrix[i][0] <= target <= matrix[i][cols - 1]:

#                 for j in range(cols):
#                     if matrix[i][j] == target:
#                         return True

#         return False






# select trim(lower(product_name)) as product_name, date_format(sale_date,'%Y-%m') as sale_date , count(product_name) as total
# from sales
# group by 1, 2
# order by 1 asc,2 asc


# select
#     lower(trim(product_name)) product_name, left(sale_date, 7) sale_date, count(sale_id) total
# from sales
# group by 1, 2
# order by 1, 2




# 1350. Students With Invalid Departments
# SELECT Students.id, Students.name FROM Students
# LEFT JOIN Departments ON (Students.department_id = Departments.id)
# WHERE Departments.id IS NULL



# 181. Employees Earning More Than Their Managers
# select a.name as 'Employee'
# from Employee as a, Employee as b
# where a.managerId = b.id
# AND a.salary > b.salary



# 176. Second Highest Salary
# SELECT
# (Select distinct Employee.salary
# from Employee
# ORDER BY Employee.salary desc
# Limit 1 offset 1) as 'SecondHighestSalary'


# 1303. Find the Team Size
# SELECT
#     employee_id,
#     (SELECT COUNT(team_id) FROM Employee e WHERE e.team_id = ee.team_id) as team_size
# FROM Employee as ee
# GROUP BY employee_id




# 39. Combination Sum
# class Solution:
#     # def __init__(self):
#     #     # self.results = []

#     def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:

#         results = []
#         def backtracker(index, remaining, comb):
#             if remaining == 0:
#                 results.append(list(comb))
#                 return
#             elif remaining < 0:
#                 return

#             for i in range(index, len(candidates)):
#                 comb.append(candidates[i])
#                 backtracker(i, remaining - candidates[i], comb)
#                 comb.pop()

#         backtracker(0, target, [])
#         return results





# 222. Count Complete Tree Nodes
# class Solution:
#     # def __init__(self):
#     #     self.count = 0

#     def countNodes(self, root: Optional[TreeNode]) -> int:

#         def helper(node):
#             if not node:
#                 return 0
#             if not node.left and not node.right:
#                 return 1
#             left = helper(node.left)
#             right = helper(node.right)

#             return 1 + left + right

#         return helper(root)



# 182. Duplicate Emails
# select Email
# from Person
# group by Email
# having count(Email) > 1



# 183. Customers Who Never Order
# select customers.name as Customers
# from customers
# left join orders on (orders.customerId = customers.id)
# where orders.customerId is null




# 125. Valid Palindrome
# class Solution:
#     def isPalindrome(self, s: str) -> bool:
#         left = 0
#         right = len(s) - 1

#         while left < right:
#             while left < right and not s[left].isalnum():
#                 left += 1
#             while left < right and not s[right].isalnum():
#                 right -= 1
#             if s[left].lower() != s[right].lower():
#                 return False
#             left += 1
#             right -= 1

#         return True



# 680. Valid Palindrome II
# class Solution:
#     def validPalindrome(self, s: str) -> bool:
#         left = 0
#         right = len(s) - 1


#         def checkPalindrome(s, l, r):
#             while l < r:
#                 if s[l] != s[r]:
#                     return False
#                 l += 1
#                 r -= 1
#             return True

#         while left < right:
#             if s[left] != s[right]:
#                 return checkPalindrome(s, left + 1, right) or checkPalindrome(s, left, right - 1)
#             left += 1
#             right -= 1

#         return True


# 387. First Unique Character in a String
# class Solution:
#     def firstUniqChar(self, s: str) -> int:
#         if len(s) == 1:
#             return 0
#         # for i in range(len(s)):
#         #     if (i + 1) in range(len(s)) and s[i] != s[i + 1]:
#         #         ch = s[i]
#         #         if s.count(ch) == 1:
#         #             return i
#         #     if i == len(s) - 1 and s[i - 1] == s[i] and s.count(s[i]) == 1:
#         #         return i
#         for i in range(len(s)):
#             if s.count(s[i]) == 1:
#                 return i


#         return -1


# 451. Sort Characters By Frequency
# class Solution:
#     def frequencySort(self, s: str) -> str:

#     # Count up the occurances.
#         counts = collections.Counter(s)

#         # Build up the string builder.
#         string_builder = []
#         for letter, freq in counts.most_common():
#             # letter * freq makes freq copies of letter.
#             # e.g. "a" * 4 -> "aaaa"
#             string_builder.append(letter * freq)
#         return "".join(string_builder)


# 49. Group Anagrams
# class Solution:
#     def groupAnagrams(self, strs: List[str]) -> List[List[str]]:

#         res = collections.defaultdict(list)


#         for word in strs:
#             res[tuple(sorted(word))].append(word)

#         return res.values()


# 242. Valid Anagram
# class Solution:
#     def isAnagram(self, s: str, t: str) -> bool:
#         return sorted(s) == sorted(t)



# 438. Find All Anagrams in a String
# class Solution:
#     def findAnagrams(self, s: str, p: str) -> List[int]:

#         res = []
#         p = sorted(p)
#         p_length = len(p)

#         for i in range(len(s)):
#             if s[i] in p and (i + p_length - 1) in range(len(s)):
#                 curr_check = s[i: i + p_length]
#                 sorted_curr_check = sorted(curr_check)
#                 if sorted_curr_check == p:
#                     res.append(i)

#         return res



# 567. Permutation in String
# class Solution:
#     def checkInclusion(self, s1: str, s2: str) -> bool:

#         s1_sorted = sorted(s1)
#         s1_length = len(s1)
#         s2_length = len(s2)

#         for i in range(s2_length):
#             if s2[i] in s1 and (i + s1_length - 1) in range(s2_length):
#                 curr = s2[i: i + s1_length]
#                 curr_sorted = sorted(curr)
#                 if curr_sorted == s1_sorted:
#                     return True

#         return False




# 1047. Remove All Adjacent Duplicates In String
# class Solution:
#     def removeDuplicates(self, s: str) -> str:

#         stack = [[s[0], 1]]

#         for index in range(1, len(s)):
#             if stack and stack[-1][0] == s[index]:
#                 stack[-1][1] += 1

#                 if stack[-1][1] == 2:
#                     stack.pop()
#             else:
#                 stack.append([s[index], 1])
#         answer = ""
#         for letter, frequency in stack:
#             answer += letter * frequency

#         return answer






# 1209. Remove All Adjacent Duplicates in String II
# class Solution:
#     def removeDuplicates(self, s: str, k: int) -> str:

#         stack = [[s[0], 1]]

#         for index in range(1, len(s)):
#             if stack and s[index] == stack[-1][0]:
#                 stack[-1][1] += 1

#                 if stack[-1][1] == k:
#                     stack.pop()
#             else:
#                 stack.append([s[index], 1])

#         answer = ""
#         for letter, count in stack:
#             answer += letter * count

#         return answer




# 1190. Reverse Substrings Between Each Pair of Parentheses
# class Solution:
#     def reverseParentheses(self, string: str) -> str:

#         stack = []

#         for character in string:
#             if character == ')':
#                 temp = ''
#                 while stack and stack[-1] != '(':
#                     temp += stack.pop()
#                 stack.pop()
#                 for letter in temp:
#                     stack.append(letter)
#             else:
#                 stack.append(character)

#         return "".join(stack)





# 1062. Longest Repeating Substring
# class Solution:
#     def search(self, L: int, n: int, S: str) -> str:
#         """
#         Search a substring of given length
#         that occurs at least 2 times.
#         @return start position if the substring exits and -1 otherwise.
#         """
#         seen = set()
#         for start in range(0, n - L + 1):
#             tmp = S[start:start + L]
#             if tmp in seen:
#                 return start
#             seen.add(tmp)
#         return -1

#     def longestRepeatingSubstring(self, S: str) -> str:
#         n = len(S)

#         # binary search, L = repeating string length
#         left, right = 1, n
#         while left <= right:
#             L = left + (right - left) // 2
#             if self.search(L, n, S) != -1:
#                 left = L + 1
#             else:
#                 right = L - 1

#         return left - 1
