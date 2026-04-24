"""
高难度实验数据集

目的: 上一轮 GSM8K 数学题 DeepSeek 初始满分，无法区分策略差异。
本数据集大幅提升难度，目标初始正确率 30%-70%，给优化留出空间。

包含:
- MATH_HARD: 竞赛级数学题 (代数/数论/概率/组合/几何)
- LOGIC_HARD: 高难度逻辑推理 (多步约束/博弈/概率直觉/悖论型)
- CODE_HARD: 算法级代码 bug (边界条件/算法错误/隐蔽 bug)
"""


# ─── MATH-Hard: 竞赛级数学 ───
# 答案格式: "#### 数字"（与 MathAdapter 兼容）

MATH_HARD_TRAINSET = [
    # 数论
    {
        "question": "Find the remainder when 2^100 is divided by 7.",
        "answer": "#### 2"
    },
    {
        "question": "How many positive integers less than 1000 are divisible by 3 but not by 5?",
        "answer": "#### 267"
    },
    {
        "question": "What is the sum of all positive divisors of 360?",
        "answer": "#### 1170"
    },
    {
        "question": "Find the last two digits of 7^2024.",
        "answer": "#### 01"
    },
    # 组合
    {
        "question": "In how many ways can 8 people be seated around a circular table? (rotations are considered the same)",
        "answer": "#### 5040"
    },
    {
        "question": "How many 4-digit numbers have digits that sum to 10?",
        "answer": "#### 219"
    },
    {
        "question": "A committee of 5 is to be formed from 6 men and 4 women. In how many ways can this be done if the committee must have at least 2 women?",
        "answer": "#### 186"
    },
    # 概率
    {
        "question": "Three dice are rolled. What is the probability that the sum is 10? Express as a fraction p/q in lowest terms, and give the value of p+q.",
        "answer": "#### 34"
    },
    {
        "question": "A bag has 5 red and 3 blue balls. Two balls are drawn without replacement. What is the probability both are red? Express as a simplified fraction numerator/denominator and give the numerator.",
        "answer": "#### 5"
    },
    # 代数
    {
        "question": "If x + 1/x = 5, what is the value of x^3 + 1/x^3?",
        "answer": "#### 110"
    },
    {
        "question": "Find the sum of all real solutions of the equation |x^2 - 4x + 3| = 3.",
        "answer": "#### 8"
    },
    {
        "question": "The roots of x^2 - 7x + k = 0 are both prime numbers. What is k?",
        "answer": "#### 10"
    },
    # 几何/面积
    {
        "question": "A right triangle has legs of length 5 and 12. What is the length of the altitude from the right angle to the hypotenuse? Express as a fraction p/q in lowest terms and give p+q.",
        "answer": "#### 73"
    },
    {
        "question": "A circle is inscribed in a right triangle with legs 6 and 8 and hypotenuse 10. What is the radius of the inscribed circle?",
        "answer": "#### 2"
    },
    # 序列
    {
        "question": "The first term of a geometric sequence is 3 and the common ratio is 2. What is the sum of the first 10 terms?",
        "answer": "#### 3069"
    },
    {
        "question": "What is the 20th term of the sequence defined by a(1)=1, a(n)=a(n-1)+2n-1 for n>=2?",
        "answer": "#### 400"
    },
    # 进阶
    {
        "question": "How many trailing zeros does 50! (50 factorial) have?",
        "answer": "#### 12"
    },
    {
        "question": "What is the greatest common divisor of 2024 and 1234?",
        "answer": "#### 2"
    },
    {
        "question": "The number 2^10 - 1 = 1023. How many positive divisors does 1023 have?",
        "answer": "#### 16"
    },
    {
        "question": "In how many ways can you make change for 25 cents using pennies (1c), nickels (5c), and dimes (10c)?",
        "answer": "#### 12"
    },
]

MATH_HARD_VALSET = [
    {
        "question": "Find the remainder when 3^50 is divided by 11.",
        "answer": "#### 1"
    },
    {
        "question": "How many positive integers n <= 100 are such that n^2 + n is divisible by 6?",
        "answer": "#### 67"
    },
    {
        "question": "If f(x) = x^2 - 3x + 2, what is f(f(0))?",
        "answer": "#### 0"
    },
    {
        "question": "A regular hexagon has side length 4. What is its area? Express as p*sqrt(3) and give p.",
        "answer": "#### 24"
    },
    {
        "question": "How many 3-element subsets of {1,2,3,...,10} have the property that no two elements are consecutive?",
        "answer": "#### 56"
    },
    {
        "question": "Five cards are drawn from a standard 52-card deck. What is the number of possible 5-card hands that contain exactly one pair? (one pair = exactly two cards of the same rank, and the other three cards all different ranks)",
        "answer": "#### 1098240"
    },
    {
        "question": "What is the least common multiple of 12, 18, and 30?",
        "answer": "#### 180"
    },
    {
        "question": "Find the sum: 1/1*2 + 1/2*3 + 1/3*4 + ... + 1/99*100. Express as a fraction p/q in lowest terms and give p.",
        "answer": "#### 99"
    },
    {
        "question": "How many trailing zeros does 100! have?",
        "answer": "#### 24"
    },
    {
        "question": "Two trains start 300 km apart and travel toward each other. Train A goes 70 km/h and Train B goes 80 km/h. A fly starts at Train A and flies at 120 km/h back and forth between the trains until they meet. How many km does the fly travel in total?",
        "answer": "#### 240"
    },
]


# ─── LOGIC-Hard: 高难度逻辑推理 ───
# 多步推理、约束满足、概率直觉、博弈论

LOGIC_HARD_TRAINSET = [
    # 概率直觉 (反直觉)
    {
        "question": "In the Monty Hall problem, you pick door 1. The host opens door 3 (which has a goat). Should you switch to door 2 or stay with door 1 to maximize your chance of winning the car? Answer with ONLY 'switch' or 'stay'.",
        "answer": "switch"
    },
    {
        "question": "You flip a fair coin 4 times. What is the probability of getting exactly 2 heads? Express as a fraction in lowest terms (e.g. 3/8).",
        "answer": "3/8"
    },
    # 约束满足
    {
        "question": "Alice, Bob, and Carol each wear a different color hat: red, blue, green. Alice says: 'I'm not wearing red.' Bob says: 'I'm not wearing blue.' Carol says: 'I'm not wearing red.' If exactly one person is lying, who is wearing the red hat? Answer with ONLY the name.",
        "answer": "Carol"
    },
    {
        "question": "Three boxes are labeled 'Apples', 'Oranges', and 'Mixed'. ALL labels are wrong. You pick one fruit from the box labeled 'Mixed' and it's an apple. What does the box labeled 'Oranges' actually contain? Answer with ONLY one word in lowercase.",
        "answer": "mixed"
    },
    {
        "question": "Five houses in a row are painted different colors. The English person lives in the red house. The Spanish person has a dog. Coffee is drunk in the green house. The green house is directly to the right of the ivory house. The person who drinks milk lives in the middle house. Who drinks water? Answer with ONLY the nationality.",
        "answer": "Norwegian"
    },
    # 数学逻辑
    {
        "question": "A king tells a prisoner: 'Make a statement. If it's true, I'll hang you. If it's false, I'll shoot you.' The prisoner says something and goes free. What logical approach did the prisoner use? Answer: the prisoner said 'You will ___ me'. Fill in the ONE missing word.",
        "answer": "shoot"
    },
    {
        "question": "100 prisoners each have a number (1-100) on their back. They can see everyone else's number but not their own. They must simultaneously guess their own number. Using a pre-agreed strategy, what is the maximum number of prisoners guaranteed to survive? Answer with ONLY the number.",
        "answer": "99"
    },
    {
        "question": "You have 12 identical-looking coins, one of which is either heavier or lighter than the rest. Using a balance scale, what is the minimum number of weighings needed to identify the odd coin AND determine if it's heavier or lighter? Answer with ONLY the number.",
        "answer": "3"
    },
    # 序列/模式
    {
        "question": "What is the next number: 1, 11, 21, 1211, 111221, ...? Answer with ONLY the number.",
        "answer": "312211"
    },
    {
        "question": "Each person in a room shakes hands with every other person exactly once. If 45 handshakes occurred, how many people are in the room? Answer with ONLY the number.",
        "answer": "10"
    },
    # 博弈论
    {
        "question": "There are 21 stones. Two players take turns removing 1, 2, or 3 stones. The player who takes the last stone wins. If you go first, how many stones should you take to guarantee a win? Answer with ONLY the number.",
        "answer": "1"
    },
    {
        "question": "A jailer tells 3 prisoners that tomorrow 2 will be freed and 1 executed, chosen at random. Prisoner A asks the jailer: 'Since at least one of B and C will be freed, tell me one who will be freed.' The jailer says 'B will be freed.' What is prisoner A's probability of being executed? Express as a fraction in lowest terms.",
        "answer": "1/3"
    },
    # 多步推理
    {
        "question": "On an island, blue-eyed people must leave on the night they discover they have blue eyes. Everyone can see others' eyes but not their own. If there are 3 blue-eyed people and a visitor says 'I see someone with blue eyes', on which night do they leave? Answer with ONLY the number.",
        "answer": "3"
    },
    {
        "question": "A snail is at the bottom of a 30-foot well. Each day it climbs 3 feet, but at night it slips back 2 feet. On which day does the snail reach the top? Answer with ONLY the number.",
        "answer": "28"
    },
    {
        "question": "You have 8 identical-looking balls. One is slightly heavier. Using a balance scale, what is the minimum number of weighings needed to find the heavy ball? Answer with ONLY the number.",
        "answer": "2"
    },
    # 逻辑谜题
    {
        "question": "A says 'B is a liar.' B says 'C is a liar.' C says 'A and B are both liars.' If exactly one of them is a truth-teller, who is it? Answer with ONLY the letter.",
        "answer": "B"
    },
    {
        "question": "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies? Answer with ONLY 'yes' or 'no'.",
        "answer": "yes"
    },
    {
        "question": "You meet two guards. One always tells the truth, one always lies. One door leads to freedom, one to death. You can ask one guard one question. What do you ask? The optimal question is: 'If I asked the OTHER guard which door leads to freedom, what would they say?' Then you choose the ___ door. Fill in: opposite",
        "answer": "opposite"
    },
    {
        "question": "In a room of 23 people, what is the approximate probability (to the nearest percent) that at least two share a birthday? Answer with ONLY the number.",
        "answer": "50"
    },
    {
        "question": "You have a 3-gallon jug and a 5-gallon jug. How do you measure exactly 4 gallons? How many total pour/fill/empty steps is the minimum? Answer with ONLY the number of steps.",
        "answer": "6"
    },
]

LOGIC_HARD_VALSET = [
    {
        "question": "Two players alternate placing non-overlapping coins of the same size on a round table. The last player who can place a coin wins. Does the first or second player have the winning strategy? Answer with ONLY 'first' or 'second'.",
        "answer": "first"
    },
    {
        "question": "A census taker asks a woman: 'How old are your three children?' She says: 'The product of their ages is 36, and the sum is the house number next door.' The census taker looks at the house number and says: 'I need more info.' She says: 'The oldest plays piano.' What are the ages? Answer as three numbers separated by commas from smallest to largest.",
        "answer": "1, 6, 6"
    },
    {
        "question": "Four people need to cross a bridge at night with one flashlight. They must cross in pairs or alone, always with the flashlight. Crossing times: A=1min, B=2min, C=5min, D=10min. A pair crosses at the slower person's speed. What is the minimum total time? Answer with ONLY the number of minutes.",
        "answer": "17"
    },
    {
        "question": "There are 1000 lockers, all closed. Person 1 opens every locker. Person 2 toggles every 2nd locker. Person 3 toggles every 3rd, and so on up to person 1000. How many lockers are open at the end? Answer with ONLY the number.",
        "answer": "31"
    },
    {
        "question": "A farmer needs to cross a river with a wolf, a goat, and a cabbage. The boat fits only the farmer and one item. The wolf will eat the goat, and the goat will eat the cabbage if left alone. What is the minimum number of boat trips (one direction = one trip) needed? Answer with ONLY the number.",
        "answer": "7"
    },
    {
        "question": "You roll two fair six-sided dice. What is the probability that the product of the two numbers is even? Express as a fraction in lowest terms.",
        "answer": "3/4"
    },
    {
        "question": "In a knockout tournament with 64 players, how many total matches are played to determine the winner? Answer with ONLY the number.",
        "answer": "63"
    },
    {
        "question": "What is the sum 1 + 2 + 3 + ... + 100? Answer with ONLY the number.",
        "answer": "5050"
    },
    {
        "question": "You have a 100-story building and 2 identical eggs. You need to find the highest floor from which an egg can be dropped without breaking. What is the minimum number of drops needed in the worst case? Answer with ONLY the number.",
        "answer": "14"
    },
    {
        "question": "There are 10 bags, each containing 10 coins. In 9 bags, each coin weighs 10g. In 1 bag, each coin weighs 9g. Using a digital scale, what is the minimum number of weighings needed to find the lighter bag? Answer with ONLY the number.",
        "answer": "1"
    },
    # 新增验证题 (扩充至20题)
    {
        "question": "Five pirates split 100 gold coins. The most senior proposes a split and everyone votes. The proposal passes if at least half agree. If rejected, the proposer is thrown overboard and the next pirate proposes. Pirates are perfectly rational: they prioritize survival first, then maximizing gold. How many coins does the most senior pirate keep? Answer with ONLY the number.",
        "answer": "98"
    },
    {
        "question": "You have 25 horses and can race 5 at a time with no stopwatch. What is the minimum number of races needed to determine the 3 fastest horses? Answer with ONLY the number.",
        "answer": "7"
    },
    {
        "question": "In a room with 50 people, at least how many people must share the same birth month? Assume exactly 12 months. Answer with ONLY the number.",
        "answer": "5"
    },
    {
        "question": "In a 3x3 magic square using numbers 1 through 9 where every row, column, and diagonal sums to 15, what number must be in the center cell? Answer with ONLY the number.",
        "answer": "5"
    },
    {
        "question": "Three perfectly logical people walk into a bar. The bartender asks: 'Does everyone here want a beer?' The first person says 'I don't know.' The second says 'I don't know.' The third says 'Yes.' How many of the three want a beer? Answer with ONLY the number.",
        "answer": "3"
    },
    {
        "question": "A game show has 100 doors. One hides a car, the rest hide goats. You pick door 1. The host, who knows what's behind each door, opens 98 other doors, all revealing goats. You may switch to the one remaining closed door. What is the probability of winning if you switch? Express as a fraction in lowest terms.",
        "answer": "99/100"
    },
    {
        "question": "A clock loses exactly 10 minutes every hour. If it is set correctly at 12:00 noon, what is the actual time when the clock first shows 10:00 PM? Answer with ONLY the time in format like 12:00 AM.",
        "answer": "12:00 AM"
    },
    {
        "question": "How many squares of any size can be found on a standard 8x8 chessboard? Answer with ONLY the number.",
        "answer": "204"
    },
    {
        "question": "A prisoner is told: 'If your statement is true, you will be hanged. If false, you will be drowned.' The prisoner makes a statement and goes free because neither punishment can be applied. He said: 'I will be ___.' Fill in the ONE missing word in lowercase.",
        "answer": "drowned"
    },
    {
        "question": "Two trains start 200 km apart heading toward each other, each traveling at 50 km/h. A bee starts at one train and flies back and forth between them at 75 km/h until the trains meet. How many km does the bee travel in total? Answer with ONLY the number.",
        "answer": "150"
    },
]


# ─── CODE-Hard: 算法级代码 Bug ───

CODE_HARD_TRAINSET = [
    {
        "question": "What is the bug in this Python code?\ndef binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n\n# Bug scenario: binary_search(list(range(2**31)), 2**31 - 1)\nAnswer with ONLY the fix in one sentence.",
        "answer": "mid = (lo + hi) // 2 can cause integer overflow in some languages; use mid = lo + (hi - lo) // 2 instead"
    },
    {
        "question": "What is the bug in this Python code?\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr if x < pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + [pivot] + quicksort(right)\n\n# Bug scenario: quicksort([3, 1, 3, 2, 3])\nAnswer with ONLY the fix in one sentence.",
        "answer": "elements equal to the pivot are lost; change right filter to x >= pivot and exclude the first pivot, or collect equals separately"
    },
    {
        "question": "What is the bug in this Python code?\ndef is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, n):\n        if n % i == 0:\n            return False\n    return True\n\n# Bug: is_prime(1000000007) takes forever\nAnswer with ONLY the fix in one sentence.",
        "answer": "only check divisors up to sqrt(n) by changing range to range(2, int(n**0.5) + 1)"
    },
    {
        "question": "What is the bug in this Python code?\ndef flatten_dict(d, parent_key='', sep='.'):\n    items = []\n    for k, v in d.items():\n        new_key = parent_key + sep + k if parent_key else k\n        if isinstance(v, dict):\n            items.extend(flatten_dict(v, new_key, sep=sep))\n        else:\n            items.append((new_key, v))\n    return dict(items)\n\n# Bug scenario: flatten_dict({'a': {'b': {'c': 1}}})\nAnswer with ONLY the fix in one sentence.",
        "answer": "the recursive call returns a dict but extend expects an iterable of tuples; use .items() on the recursive result"
    },
    {
        "question": "What is the bug in this Python code?\ndef lru_cache(capacity):\n    cache = {}\n    order = []\n    def get(key):\n        if key in cache:\n            order.remove(key)\n            order.append(key)\n            return cache[key]\n        return -1\n    def put(key, value):\n        if key in cache:\n            order.remove(key)\n        elif len(cache) >= capacity:\n            oldest = order.pop(0)\n            del cache[oldest]\n        cache[key] = value\n        order.append(key)\n    return get, put\n\n# Bug: O(n) remove operation makes this inefficient\nAnswer with ONLY the fix in one sentence.",
        "answer": "use an OrderedDict instead of a list for O(1) removal and reordering"
    },
    {
        "question": "What is the bug in this Python code?\nimport threading\ncounter = 0\ndef increment():\n    global counter\n    for _ in range(100000):\n        counter += 1\n\nthreads = [threading.Thread(target=increment) for _ in range(10)]\nfor t in threads: t.start()\nfor t in threads: t.join()\nprint(counter)  # Expected: 1000000\nAnswer with ONLY the fix in one sentence.",
        "answer": "counter += 1 is not atomic; use a threading.Lock to synchronize access to the shared counter"
    },
    {
        "question": "What is the bug in this Python code?\ndef merge_intervals(intervals):\n    intervals.sort()\n    merged = [intervals[0]]\n    for start, end in intervals[1:]:\n        if start <= merged[-1][1]:\n            merged[-1][1] = end\n        else:\n            merged.append([start, end])\n    return merged\n\n# Bug scenario: merge_intervals([[1,10],[2,3],[4,6]])\nAnswer with ONLY the fix in one sentence.",
        "answer": "when merging overlapping intervals, use max(merged[-1][1], end) instead of just end to handle contained intervals"
    },
    {
        "question": "What is the bug in this Python code?\ndef topological_sort(graph):\n    visited = set()\n    result = []\n    def dfs(node):\n        if node in visited:\n            return\n        visited.add(node)\n        for neighbor in graph.get(node, []):\n            dfs(neighbor)\n        result.append(node)\n    for node in graph:\n        dfs(node)\n    return result\n\n# Bug: doesn't detect cycles\nAnswer with ONLY the fix in one sentence.",
        "answer": "add a separate 'in_progress' set to detect back edges and raise an error when a cycle is found"
    },
    {
        "question": "What is the bug in this Python code?\ndef deepcopy_list(lst):\n    return lst[:]\n\n# Bug scenario: a = [[1,2],[3,4]]; b = deepcopy_list(a); b[0][0] = 99; print(a)\nAnswer with ONLY the fix in one sentence.",
        "answer": "lst[:] only creates a shallow copy; use copy.deepcopy(lst) or [x[:] for x in lst] for nested lists"
    },
    {
        "question": "What is the bug in this Python code?\ndef memoize(func):\n    cache = {}\n    def wrapper(*args):\n        if args not in cache:\n            cache[args] = func(*args)\n        return cache[args]\n    return wrapper\n\n@memoize\ndef fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)\n\n# Bug scenario: fib(1000) causes RecursionError\nAnswer with ONLY the fix in one sentence.",
        "answer": "increase sys.setrecursionlimit or use iterative bottom-up approach to build the cache without deep recursion"
    },
]

CODE_HARD_VALSET = [
    {
        "question": "What is the bug in this Python code?\ndef find_duplicates(lst):\n    seen = set()\n    duplicates = set()\n    for item in lst:\n        if item in seen:\n            duplicates.add(item)\n        seen.add(item)\n    return list(duplicates)\n\n# Bug scenario: find_duplicates([1, [2], [2]])\nAnswer with ONLY the fix in one sentence.",
        "answer": "lists are unhashable and cannot be added to sets; convert to tuples first or use a different approach"
    },
    {
        "question": "What is the bug in this Python code?\ndef count_islands(grid):\n    rows, cols = len(grid), len(grid[0])\n    count = 0\n    def dfs(r, c):\n        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == 0:\n            return\n        grid[r][c] = 0\n        dfs(r+1, c); dfs(r-1, c); dfs(r, c+1); dfs(r, c-1)\n    for r in range(rows):\n        for c in range(cols):\n            if grid[r][c] == 1:\n                count += 1\n                dfs(r, c)\n    return count\n\n# Bug scenario: large grid (500x500) causes RecursionError\nAnswer with ONLY the fix in one sentence.",
        "answer": "replace recursive DFS with iterative BFS using a queue to avoid hitting Python's recursion limit on large grids"
    },
    {
        "question": "What is the bug in this Python code?\ndef parse_json_numbers(text):\n    import json\n    data = json.loads(text)\n    total = 0\n    def walk(obj):\n        nonlocal total\n        if isinstance(obj, (int, float)):\n            total += obj\n        elif isinstance(obj, dict):\n            for v in obj.values():\n                walk(v)\n        elif isinstance(obj, list):\n            for item in obj:\n                walk(item)\n    walk(data)\n    return total\n\n# Bug scenario: parse_json_numbers('{\"a\": 1e308, \"b\": 1e308}')\nAnswer with ONLY the fix in one sentence.",
        "answer": "summing very large floats can cause overflow to inf; use decimal.Decimal or check for inf/nan values"
    },
    {
        "question": "What is the bug in this Python code?\ndef power_mod(base, exp, mod):\n    result = 1\n    base = base % mod\n    while exp > 0:\n        if exp % 2 == 1:\n            result = result * base % mod\n        exp = exp // 2\n        base = base * base % mod\n    return result\n\n# Bug scenario: power_mod(5, 0, 1)\nAnswer with ONLY the fix in one sentence.",
        "answer": "when mod is 1, any number mod 1 is 0, but the function returns 1; add a check if mod == 1: return 0"
    },
    {
        "question": "What is the bug in this Python code?\nfrom collections import defaultdict\ndef detect_cycle(edges):\n    graph = defaultdict(list)\n    for u, v in edges:\n        graph[u].append(v)\n    visited = set()\n    def dfs(node):\n        if node in visited:\n            return True\n        visited.add(node)\n        for nei in graph[node]:\n            if dfs(nei):\n                return True\n        return False\n    return any(dfs(n) for n in graph)\n\n# Bug scenario: detects false cycles in DAGs\nAnswer with ONLY the fix in one sentence.",
        "answer": "visited should be split into 'in_stack' (current path) and 'done' (finished) sets; only report a cycle if revisiting a node in the current DFS path"
    },
]
