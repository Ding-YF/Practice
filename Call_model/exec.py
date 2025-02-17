
def max_sum_of_two(a, b, c):
    return sum(sorted([a, b, c])[-2:])

# 示例输入
num1 = int(input("请输入第一个数: "))
num2 = int(input("请输入第二个数: "))
num3 = int(input("请输入第三个数: "))

# 计算并输出结果
result = max_sum_of_two(num1, num2, num3)
print("最大两个数的和为:", result)