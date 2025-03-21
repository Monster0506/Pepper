from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic = dict()
        for i in range(len(nums)):
            value = nums[i]
            complement = target - value
            if complement in dic:
                return [i, dic[complement]]
            dic[value] = i
        return []


def test(case):
    solution = Solution()
    for x in case:
        assert solution.twoSum(case[x][0], case[x][1]) == x[0] or x[1]
        print("passed: " + str(case[x]))


def createTestCase(target, nums, solution):
    solutions = tuple(
        [tuple([solution[0], solution[1]]), tuple([solution[1], solution[0]])]
    )

    return [solutions, [nums, target]]


def createDictionary(keys, a):
    for key in keys:
        a[key[0]] = key[1]


def runTests(tests):
    keys = list()
    a = dict()
    for test_ in tests:
        key = createTestCase(test_[0], test_[1], test_[2])
        keys.append(key)

    createDictionary(keys, a)
    test(a)


cases = [[9, [15, 2, 7, 11, 15], [1, 2]], [9, [2, 7, 11, 15], [0, 1]]]

runTests(cases)
