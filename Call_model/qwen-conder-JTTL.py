def solve_chicken_rabbit(total_heads, total_legs):
    # 鸡和兔子的数量
    for chickens in range(total_heads + 1):
        rabbits = total_heads - chickens
        if 2 * chickens + 4 * rabbits == total_legs:
            return chickens, rabbits
    return None

def main():
    try:
        total_heads = int(input("请输入总头数: "))
        total_legs = int(input("请输入总腿数: "))

        result = solve_chicken_rabbit(total_heads, total_legs)

        if result is not None:
            chickens, rabbits = result
            print(f"鸡的数量: {chickens}")
            print(f"兔子的数量: {rabbits}")
        else:
            print("输入的参数无解")

    except ValueError:
        print("输入无效，请输入整数")

if __name__ == "__main__":
    main()