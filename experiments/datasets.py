"""
实验数据集模块

包含:
- GSM8K 数学推理数据集（精选子集）
- 逻辑推理数据集（扩展版）
- 代码修复数据集（扩展版）
"""


# ─── GSM8K 数学推理数据集 ───
# 从 GSM8K 中精选的有代表性的题目
# 格式: {"question": "...", "answer": "#### 数字"}

GSM8K_TRAINSET = [
    {
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every remaining duck egg at the farmers' market for $2 per egg. How much in dollars does she make every day at the farmers' market?",
        "answer": "#### 18"
    },
    {
        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "answer": "#### 3"
    },
    {
        "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
        "answer": "#### 70000"
    },
    {
        "question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
        "answer": "#### 540"
    },
    {
        "question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. If she fetches a total of 65 cups of feed for the final meal of the day, how many chickens does Wendi have?",
        "answer": "#### 35"
    },
    {
        "question": "Kylar went to the store to get water and some tissues. He bought 4 tissues at $2 each and a water bottle that costs $1 less than the total cost of the tissues. How much did Kylar spend in all?",
        "answer": "#### 15"
    },
    {
        "question": "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. If Seattle has 20 sheep, how many sheep do Toulouse, Charleston, and Seattle have combined?",
        "answer": "#### 260"
    },
    {
        "question": "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. After the restart, she has to restart the download from the beginning. How long does it take her to download the file in minutes?",
        "answer": "#### 220"
    },
    {
        "question": "John drives for 3 hours at a speed of 60 mph and then turns around because he realizes he forgot something very important at home. He tries to get home in 4 hours but spends the first 2 hours in standstill traffic. He spends the rest of the time driving at 30 mph. How far is he from home at the end of 4 hours?",
        "answer": "#### 120"
    },
    {
        "question": "Eliza's rate per hour for the first 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week?",
        "answer": "#### 460"
    },
    {
        "question": "A new program had 60 downloads in the first month. The number of downloads in the second month was three times as many as in the first month, but then the third month saw a 30% drop from the second month. How many downloads did the program have in total over the three months?",
        "answer": "#### 366"
    },
    {
        "question": "Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per dozen, and 6 dozen mini cheesecakes which cost $55 per dozen for a birthday party. How much was the total cost?",
        "answer": "#### 694"
    },
    {
        "question": "Carlos is planting a lemon tree. The tree will cost $90 to plant. Each year it will grow 7 lemons, which he can sell for $1.5 each. It costs $3 a year to water and feed the tree. How many years will it take before he starts making money on the lemon tree?",
        "answer": "#### 13"
    },
    {
        "question": "Marissa is hiking a 12-mile trail. She took 1 hour to walk the first 4 miles, then she took 30 minutes to walk the next 2 miles, and twice as long for the final stretch. How many minutes in total did it take her to complete the trail?",
        "answer": "#### 150"
    },
    {
        "question": "Billy sells DVDs. He has 8 customers on Tuesday. His first 3 customers buy one DVD each. His next 2 customers buy 2 DVDs each. His last 3 customers don't buy any DVDs. How many DVDs did Billy sell on Tuesday?",
        "answer": "#### 7"
    },
    {
        "question": "A candle melts by 2 centimeters every hour that it burns. How many centimeters shorter will a candle be after burning from 1:00 PM to 5:00 PM?",
        "answer": "#### 8"
    },
    {
        "question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple combined. How many flowers does Mark have in his garden?",
        "answer": "#### 35"
    },
    {
        "question": "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many slices does he eat that day?",
        "answer": "#### 48"
    },
    {
        "question": "Ken created a care package to send to his brother, who lives 100 miles away. Ken placed the box on a scale, and it weighed 2 pounds. He then added 8 cans of beans (each 2 pounds), 3 bottles of water (each 3 pounds), and twice as many bags of chips as bottles of water (each 1 pound). What is the weight, in pounds, of Ken's care package?",
        "answer": "#### 33"
    },
    {
        "question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 on a blouse, $46 on a suit, $38 on new shoes, and $11 on a clutch purse. She also bought a belt for $18. How much money does Alexis have left from her budget?",
        "answer": "#### 57"
    },
]

GSM8K_VALSET = [
    {
        "question": "Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by her hourly wage plus 1/2 of her hourly wage. If she works 10 hours every day for 5 days, how much money does she make?",
        "answer": "#### 990"
    },
    {
        "question": "A merchant buys pencils from a supplier. Each pencil costs the merchant $0.20. The merchant then sells each pencil for $0.50. What is the merchant's profit if he sells 100 pencils?",
        "answer": "#### 30"
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "answer": "#### 10"
    },
    {
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "answer": "#### 5"
    },
    {
        "question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
        "answer": "#### 42"
    },
    {
        "question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        "answer": "#### 624"
    },
    {
        "question": "Mark has a basket of 36 apples. He gave 1/3 of the apples to his sister, and then he ate 3 himself. How many apples does Mark have left?",
        "answer": "#### 21"
    },
    {
        "question": "A farmer has a field that is 300 feet long and 500 feet wide. He wants to plant corn on 80% of the field. How many square feet of the field will be used for corn?",
        "answer": "#### 120000"
    },
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "#### 72"
    },
    {
        "question": "Tim has 30 less apples than Martha, and Harry has half as many apples as Tim. If Martha has 68 apples, how many apples does Harry have?",
        "answer": "#### 19"
    },
]


# ─── 逻辑推理数据集（扩展版）───

LOGIC_TRAINSET_EXTENDED = [
    {"question": "There are 5 people in a room. Each shakes hands with every other person exactly once. How many handshakes occur? Answer with ONLY the number.", "answer": "10"},
    {"question": "If today is Wednesday, what day will it be 100 days from now? Answer with ONLY the day name in lowercase.", "answer": "friday"},
    {"question": "A farmer has 17 sheep. All but 9 die. How many sheep are left? Answer with ONLY the number.", "answer": "9"},
    {"question": "How many months have 28 days? Answer with ONLY the number.", "answer": "12"},
    {"question": "If you have a bowl with six apples and you take away four, how many do you have? Answer with ONLY the number.", "answer": "4"},
    {"question": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in cents? Answer with ONLY the number.", "answer": "5"},
    {"question": "If there are 3 apples and you take away 2, how many apples do YOU have? Answer with ONLY the number.", "answer": "2"},
    {"question": "What is the next number in the sequence: 1, 1, 2, 3, 5, 8, 13, ...? Answer with ONLY the number.", "answer": "21"},
    {"question": "How many times can you subtract 5 from 25? Answer with ONLY the number.", "answer": "1"},
    {"question": "If a doctor gives you 3 pills and tells you to take one every 30 minutes, how many minutes will it take to finish all pills? Answer with ONLY the number.", "answer": "60"},
    # 新增题目
    {"question": "There are 8 people at a party. If everyone shakes hands with everyone else exactly once, how many handshakes are there? Answer with ONLY the number.", "answer": "28"},
    {"question": "If today is Monday, what day was it 50 days ago? Answer with ONLY the day name in lowercase.", "answer": "saturday"},
    {"question": "A train leaves station A at 9:00 AM traveling at 60 km/h. Another train leaves station B (300 km away) at the same time traveling toward station A at 40 km/h. At what time will they meet? Answer with ONLY the time in format like 12:00 PM.", "answer": "12:00 PM"},
    {"question": "You have two coins that total 30 cents. One of them is not a nickel. What are the two coins? Answer with: quarter and nickel", "answer": "quarter and nickel"},
    {"question": "Three consecutive even numbers add up to 48. What is the smallest number? Answer with ONLY the number.", "answer": "14"},
    {"question": "If 5 machines take 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets? Answer with ONLY the number.", "answer": "5"},
    {"question": "A lily pad doubles in size every day. If it takes 48 days to cover the lake completely, on what day does it cover half the lake? Answer with ONLY the number.", "answer": "47"},
    {"question": "If you count from 1 to 100, how many times does the digit 7 appear? Answer with ONLY the number.", "answer": "20"},
    {"question": "A bookshelf has 5 shelves. Each shelf holds 8 books. If 13 books are removed, how many books remain? Answer with ONLY the number.", "answer": "27"},
    {"question": "What comes next in the pattern: 2, 6, 12, 20, 30, ...? Answer with ONLY the number.", "answer": "42"},
]

LOGIC_VALSET_EXTENDED = [
    {"question": "A clock strikes 6 in 5 seconds. How many seconds will it take to strike 12? Answer with ONLY the number.", "answer": "11"},
    {"question": "How many 9s are there between 1 and 100? Answer with ONLY the number.", "answer": "20"},
    {"question": "If you rearrange the letters 'CIFAIPC' you get the name of a(n)? Answer with ONLY the word in lowercase.", "answer": "pacific"},
    {"question": "Two fathers and two sons go fishing. They each catch one fish. They catch 3 fish total. How is this possible? Answer: because there are actually ___ people (grandfather, father, son). Fill in the number ONLY.", "answer": "3"},
    {"question": "What is half of 2+2? Answer with ONLY the number.", "answer": "3"},
    {"question": "If you have 6 oranges and give half to a friend, how many do you have? Answer with ONLY the number.", "answer": "3"},
    # 新增验证题
    {"question": "If there are 12 fish and half of them drown, how many are left? Answer with ONLY the number.", "answer": "12"},
    {"question": "Some months have 30 days, some have 31. How many months have 29 days? Answer with ONLY the number.", "answer": "12"},
    {"question": "If you have only one match and you enter a dark room with an oil lamp, a newspaper, and kindling wood, what do you light first? Answer with ONLY the item in lowercase.", "answer": "match"},
    {"question": "A truck driver is going down a one-way street the wrong way. A police officer sees him but doesn't stop him. Why? Answer: because the truck driver was ___. Fill in ONE word in lowercase.", "answer": "walking"},
    # 新增验证题 (扩充至20题)
    {"question": "If you overtake the person in second place in a race, what position are you now in? Answer with ONLY the number.", "answer": "2"},
    {"question": "A drawer has 10 black socks and 10 white socks, all mixed up. Without looking, what is the minimum number of socks you must pull out to guarantee a matching pair? Answer with ONLY the number.", "answer": "3"},
    {"question": "I am an odd number. Remove one letter from my English name and I become even. What number am I? Answer with ONLY the number.", "answer": "7"},
    {"question": "If you divide 30 by half and add 10, what do you get? Answer with ONLY the number.", "answer": "70"},
    {"question": "Forward I am heavy, backward I am not. What am I? Answer with ONLY the word in lowercase.", "answer": "ton"},
    {"question": "What is the angle in degrees between the hour hand and the minute hand of a clock at exactly 3:30? Answer with ONLY the number.", "answer": "75"},
    {"question": "If 5 cats catch 5 mice in 5 minutes, how many cats are needed to catch 100 mice in 100 minutes? Answer with ONLY the number.", "answer": "5"},
    {"question": "A brick weighs 1 kilogram plus half a brick. How much does the brick weigh in kilograms? Answer with ONLY the number.", "answer": "2"},
    {"question": "Two mothers and two daughters go shopping. They each buy one hat, yet only 3 hats are bought. How many people are there? Answer with ONLY the number.", "answer": "3"},
    {"question": "If you write all integers from 1 to 20, how many times do you write the digit 1? Answer with ONLY the number.", "answer": "12"},
]


# ─── 代码修复数据集（扩展版）───

CODE_FIX_TRAINSET = [
    {"question": "What is the bug in this Python code?\ndef factorial(n):\n    result = 0\n    for i in range(1, n+1):\n        result *= i\n    return result\nAnswer with ONLY the fix in one sentence.", "answer": "result should be initialized to 1 instead of 0"},
    {"question": "What is the bug in this Python code?\ndef is_palindrome(s):\n    return s == s.reverse()\nAnswer with ONLY the fix in one sentence.", "answer": "strings don't have a reverse method, use s[::-1] instead"},
    {"question": "What is the bug in this Python code?\ndef average(numbers):\n    return sum(numbers) / len(numbers)\naverage([])\nAnswer with ONLY the fix in one sentence.", "answer": "add a check for empty list to avoid division by zero"},
    {"question": "What is the bug in this Python code?\ndef find_max(lst):\n    max_val = 0\n    for x in lst:\n        if x > max_val:\n            max_val = x\n    return max_val\nfind_max([-1, -5, -3])\nAnswer with ONLY the fix in one sentence.", "answer": "initialize max_val to float('-inf') or lst[0] instead of 0"},
    {"question": "What is the bug in this Python code?\ndef count_vowels(s):\n    vowels = 'aeiou'\n    count = 0\n    for char in s:\n        if char in vowels:\n            count += 1\n    return count\ncount_vowels('HELLO')\nAnswer with ONLY the fix in one sentence.", "answer": "convert char to lowercase before checking or add uppercase vowels"},
    {"question": "What is the bug in this Python code?\ndef remove_duplicates(lst):\n    for item in lst:\n        if lst.count(item) > 1:\n            lst.remove(item)\n    return lst\nAnswer with ONLY the fix in one sentence.", "answer": "don't modify a list while iterating over it, use a set or new list instead"},
    # 新增
    {"question": "What is the bug in this Python code?\ndef swap(a, b):\n    a = b\n    b = a\n    return a, b\nAnswer with ONLY the fix in one sentence.", "answer": "use a temporary variable or tuple unpacking a, b = b, a"},
    {"question": "What is the bug in this Python code?\ndef sum_list(lst):\n    total = 0\n    for i in range(len(lst)):\n        total += lst[i+1]\n    return total\nAnswer with ONLY the fix in one sentence.", "answer": "change lst[i+1] to lst[i] to avoid IndexError"},
    {"question": "What is the bug in this Python code?\ndef is_even(n):\n    if n % 2 = 0:\n        return True\n    return False\nAnswer with ONLY the fix in one sentence.", "answer": "use == for comparison instead of = which is assignment"},
    {"question": "What is the bug in this Python code?\ndef greet(name, greeting='Hello'):\n    return greeting + ' ' + name\ngreet(greeting='Hi')\nAnswer with ONLY the fix in one sentence.", "answer": "name parameter is missing in the call, it is required"},
]

CODE_FIX_VALSET = [
    {"question": "What is the bug in this Python code?\ndef binary_search(arr, target):\n    low, high = 0, len(arr)\n    while low < high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid\n        else:\n            high = mid\n    return -1\nAnswer with ONLY the fix in one sentence.", "answer": "low should be updated to mid + 1 to avoid infinite loop"},
    {"question": "What is the bug in this Python code?\ndef flatten(nested_list):\n    result = []\n    for item in nested_list:\n        if type(item) == list:\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result\nAnswer with ONLY the fix in one sentence.", "answer": "use isinstance(item, list) instead of type(item) == list to handle subclasses"},
    {"question": "What is the bug in this Python code?\ndef fib(n):\n    if n == 0: return 0\n    if n == 1: return 1\n    return fib(n-1) + fib(n-2)\nfib(50)\nAnswer with ONLY the fix in one sentence.", "answer": "add memoization or use iterative approach to avoid exponential time complexity"},
    {"question": "What is the bug in this Python code?\ndef power(base, exp):\n    result = 1\n    for i in range(exp):\n        result *= base\n    return result\npower(2, -3)\nAnswer with ONLY the fix in one sentence.", "answer": "handle negative exponents by returning 1/power(base, -exp)"},
    {"question": "What is the bug in this Python code?\ndef merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    return result\nAnswer with ONLY the fix in one sentence.", "answer": "append remaining elements of a and b after the while loop"},
]
