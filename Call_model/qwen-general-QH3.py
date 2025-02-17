def main():
    # 用户输入三个整数
    a = int(input("Enter first number: "))
    b = int(input("Enter second number: "))
    c = int(input("Enter third number: "))

    # 调用函数计算结果
    result = sum_of_max_and_min(a, b, c)

    # 输出结果
    print(f"The sum of the largest and smallest numbers ({max(a, b, c)} and {min(a, b, c)}) is: {result}")

# 定义求和函数
def sum_of_max_and_min(a, b, c):
    max_val = max(a, b, c)
    min_val = min(a, b, c)
    return max_val + min_val

# 运行主函数
if __name__ == "__main__":
    main()