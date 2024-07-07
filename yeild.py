def count_up_to(max):
    count = 0
    while count < max:
        yield count
        count += 1

# ジェネレータ関数を使用
counter = count_up_to(5)
print('nextでジェネレータ関数を1回呼び出す')
print(next(counter))

print('forで1回だけ回す')
for number in counter:
    print(number)
    break

print('forで最後まで回す')
for number in counter:
    print(number)
