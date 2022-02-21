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
