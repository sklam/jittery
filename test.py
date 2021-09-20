from jittery.core import translate
from prettyprinter import pprint

def foo(x):
    if 4 > x > 1:
        x += 1
    elif x < 1 or x > 4:
        x -= 1
        try:
            while x:
                while x > 3:
                    x -= 1
                else:
                    x += 1
        except:
            print("HAHA")
    else:
        x = 0
        for i in range(4):
            x += 1
            for j in range(i):
                with afda:
                    if x == 0:
                        break
    return x

# def foo(x):
#     y = 0
#     if x > 3:
#         y = 1
#     elif x < 3:
#         y = 2
#     else:
#         y = 3

#     if y == 1:
#         x = 3
#     else:
#         x = 4

#     return x, y

def foo(x):
    if 4 > x > 1:
        x += 1
    elif x < 1 or x > 4:
        x -= 1
        while x:
            while x > 3:
                x -= 1
            else:
                x += 1
    else:
        x = 0
        for i in range(4):
            x += 1
            for j in range(i):
                if x == 0:
                    break
    return x



interests = translate(foo.__code__)
pprint(interests)
