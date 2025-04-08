import random

numbers = '123456789'
ops = '+ - *'.split(' ')


def make_basic(hard):
    expr = random.choice(numbers)
    for i in range(hard):
        expr = f'{expr}{random.choice(ops)}{random.choice(numbers)}'
    return expr


def make_full(hard):
    while True:
        try:
            expr = make_basic(hard)
            for i in range(1):  #random.randint(0, hard)):
                expr = f'({expr}){random.choice(ops)}({make_basic(hard)})'
            answ = eval(expr)
            return expr, answ
        except:
            pass


def gentokens(ntok):
    alle = 'X'
    while len(alle) < ntok:
        e, a = make_full(2)
        print(e, '=', a)
        alle = alle + e
    return [ord(_) for _ in alle]


def make_block(nrows):
    import numpy as np
    lines = []
    for i in range(nrows):
        e, a = make_full(2)
        s = f'{e:>22}={a:<9}'
        # print(s)
        lines.append([ord(_) for _ in s])
    lines = np.array(lines)
    # print(lines.shape)
    return lines
    


if __name__ == '__main__':
    import numpy as np
    lines = []
    for i in range(32):
        e, a = make_full(2)
        s = f'{e:>22}={a:<9}'
        print(s)
        lines.append([ord(_) for _ in s])
    lines = np.array(lines)
    print(lines.shape)
